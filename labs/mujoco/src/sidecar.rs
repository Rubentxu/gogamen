//! MuJoCo backend as an optional Python sidecar (feature `mujoco-sidecar`).
//!
//! Spawns `python3 -P -B -c <embedded script>` (override the interpreter with the
//! `GOGAMEN_PYTHON` env var) and speaks JSON lines over stdin/stdout. The script uses
//! `mj_getState`/`mj_setState` with `mjSTATE_INTEGRATION`.
//!
//! Note: `python3 -I` is intentionally not used because isolated mode also disables the
//! per-user site-packages where `pip install --user mujoco` places the module.

use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::json::{self, Json, fmt_arr};
use crate::port::{ContinuousPhysicsEnginePort, Dims, EngineSnapshot, PortError, check_finite};

const SCRIPT: &str = include_str!("../sidecar/mobile_robot.py");
const TAG: &str = "mujoco-sidecar";

fn python() -> String {
    std::env::var("GOGAMEN_PYTHON").unwrap_or_else(|_| "python3".into())
}

/// Check whether `python3` with the `mujoco` module is usable. `Err` carries a skip reason.
pub fn probe() -> Result<String, String> {
    let out = Command::new(python())
        .args(["-P", "-B", "-c", "import mujoco; print(mujoco.__version__)"])
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("cannot run {}: {e}", python()))?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        Err(format!(
            "python module `mujoco` unavailable: {}",
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .last()
                .unwrap_or("")
        ))
    }
}

/// Maximum bytes of sidecar stderr retained (the tail is kept; the pipe is always drained).
const STDERR_CAP: usize = 4096;

/// Timing knobs of the sidecar client.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SidecarConfig {
    /// Deadline for each request/response after initialisation (default 30 s).
    pub call_timeout: Duration,
    /// Deadline for the `init` handshake, which imports MuJoCo (default 60 s).
    pub init_timeout: Duration,
    /// Grace for the child to exit after `quit` before it is killed (default 2 s).
    pub shutdown_grace: Duration,
}

impl Default for SidecarConfig {
    fn default() -> Self {
        Self {
            call_timeout: Duration::from_secs(30),
            init_timeout: Duration::from_secs(60),
            shutdown_grace: Duration::from_secs(2),
        }
    }
}

/// MuJoCo sidecar engine.
pub struct MujocoSidecarEngine {
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Receiver<String>,
    stderr_tail: Arc<Mutex<Vec<u8>>>,
    cfg: SidecarConfig,
    closed: Option<String>,
    dims: Dims,
    dt: f64,
    version: String,
}

impl MujocoSidecarEngine {
    /// Spawn the sidecar and initialise the model (default timeouts).
    pub fn spawn() -> Result<Self, PortError> {
        Self::spawn_with(SidecarConfig::default())
    }

    /// Spawn the sidecar and initialise the model with explicit timeouts.
    pub fn spawn_with(cfg: SidecarConfig) -> Result<Self, PortError> {
        let mut cmd = Command::new(python());
        cmd.args(["-P", "-B", "-c", SCRIPT]);
        let mut me = Self::from_command(cmd, cfg)?;
        me.init()?;
        Ok(me)
    }

    /// Spawn an arbitrary JSON-lines child (stdin/stdout/stderr are piped by this call) without
    /// the `init` handshake. Used with fake sidecars in tests; `dims` stay zero.
    pub fn from_command(mut cmd: Command, cfg: SidecarConfig) -> Result<Self, PortError> {
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| PortError::Backend(format!("spawn failed: {e}")))?;
        let (Some(stdin), Some(stdout), Some(mut stderr)) =
            (child.stdin.take(), child.stdout.take(), child.stderr.take())
        else {
            let _ = child.kill();
            let _ = child.wait();
            return Err(PortError::Backend("missing stdio pipe".into()));
        };
        let (tx, lines) = mpsc::channel();
        thread::spawn(move || {
            let mut r = BufReader::new(stdout);
            loop {
                let mut buf = String::new();
                match r.read_line(&mut buf) {
                    Ok(0) | Err(_) => break, // dropping `tx` signals EOF
                    Ok(_) => {
                        if tx.send(buf).is_err() {
                            break;
                        }
                    }
                }
            }
        });
        let stderr_tail = Arc::new(Mutex::new(Vec::new()));
        let tail = Arc::clone(&stderr_tail);
        thread::spawn(move || {
            let mut chunk = [0u8; 1024];
            while let Ok(n) = stderr.read(&mut chunk) {
                if n == 0 {
                    break;
                }
                if let Ok(mut t) = tail.lock() {
                    t.extend_from_slice(&chunk[..n]);
                    let over = t.len().saturating_sub(STDERR_CAP);
                    t.drain(..over);
                }
            }
        });
        Ok(Self {
            child,
            stdin: Some(stdin),
            lines,
            stderr_tail,
            cfg,
            closed: None,
            dims: Dims {
                nq: 0,
                nv: 0,
                nu: 0,
                na: 0,
            },
            dt: 0.0,
            version: String::new(),
        })
    }

    fn init(&mut self) -> Result<(), PortError> {
        let me = self;
        let timeout = me.cfg.init_timeout;
        let r = me.request("{\"cmd\":\"init\"}", timeout)?;
        let n = |k: &str| r.get(k).and_then(Json::as_f64).map(|x| x as usize);
        me.dims = Dims {
            nq: n("nq").unwrap_or(0),
            nv: n("nv").unwrap_or(0),
            nu: n("nu").unwrap_or(0),
            na: n("na").unwrap_or(0),
        };
        me.dt = r.get("dt").and_then(Json::as_f64).unwrap_or(0.0);
        me.version = r
            .get("mujoco")
            .and_then(Json::as_str)
            .unwrap_or("?")
            .to_string();
        Ok(())
    }

    /// OS pid of the sidecar process.
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Last (bounded) bytes the sidecar wrote to stderr, lossily decoded.
    pub fn stderr_tail(&self) -> String {
        self.stderr_tail
            .lock()
            .map(|t| String::from_utf8_lossy(&t).into_owned())
            .unwrap_or_default()
    }

    /// Send one JSON line and read one JSON-line reply (`ok:true` required), using the
    /// per-call timeout. After a timeout/EOF/IO failure the engine is closed for good.
    pub fn call(&mut self, line: &str) -> Result<Json, PortError> {
        self.request(line, self.cfg.call_timeout)
    }

    /// Stop the sidecar now (idempotent): `quit`, bounded grace, kill, reap.
    pub fn close(&mut self) {
        if self.closed.is_none() {
            self.closed = Some("closed by host".into());
        }
        self.shutdown();
    }

    fn shutdown(&mut self) {
        if let Some(mut stdin) = self.stdin.take() {
            let _ = writeln!(stdin, "{{\"cmd\":\"quit\"}}");
            let _ = stdin.flush();
        } // stdin dropped: EOF also ends a well-behaved sidecar
        let deadline = Instant::now() + self.cfg.shutdown_grace;
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => return,
                Ok(None) if Instant::now() < deadline => {
                    thread::sleep(Duration::from_millis(5));
                }
                _ => break,
            }
        }
        let _ = self.child.kill();
        let _ = self.child.wait(); // SIGKILL cannot be ignored: bounded
    }

    fn fail(&mut self, why: &str) {
        self.closed = Some(why.to_string());
        self.shutdown();
    }

    /// MuJoCo version reported by the sidecar.
    pub fn mujoco_version(&self) -> &str {
        &self.version
    }

    fn request(&mut self, line: &str, timeout: Duration) -> Result<Json, PortError> {
        if let Some(why) = &self.closed {
            return Err(PortError::Closed(why.clone()));
        }
        let sent = match self.stdin.as_mut() {
            Some(w) => writeln!(w, "{line}").and_then(|()| w.flush()),
            None => Err(std::io::Error::other("stdin closed")),
        };
        if let Err(e) = sent {
            self.fail("sidecar stdin write failed");
            return Err(PortError::Backend(format!("write: {e}")));
        }
        let buf = match self.lines.recv_timeout(timeout) {
            Ok(b) => b,
            Err(RecvTimeoutError::Timeout) => {
                self.fail("sidecar timed out and was killed");
                return Err(PortError::Timeout(format!(
                    "no reply within {timeout:?}; sidecar killed"
                )));
            }
            Err(RecvTimeoutError::Disconnected) => {
                self.fail("sidecar closed stdout");
                return Err(PortError::Backend("sidecar closed stdout".into()));
            }
        };
        let v = match json::parse(buf.trim()) {
            Ok(v) => v,
            Err(e) => {
                // Stream is no longer trustworthy (possible desync): stop the sidecar.
                self.fail("sidecar sent invalid JSON");
                return Err(PortError::Backend(format!("json: {e}")));
            }
        };
        if v.get("ok").and_then(Json::as_bool) != Some(true) {
            let msg = v.get("error").and_then(Json::as_str).unwrap_or("unknown");
            return Err(PortError::Backend(msg.to_string()));
        }
        Ok(v)
    }

    fn get(&mut self) -> Result<Json, PortError> {
        self.call("{\"cmd\":\"get\"}")
    }

    fn vec_field(r: &Json, k: &str) -> Result<Vec<f64>, PortError> {
        r.get(k)
            .and_then(Json::as_f64_vec)
            .ok_or_else(|| PortError::Backend(format!("missing field {k}")))
    }

    /// Deliberately incomplete restore (qpos/qvel only; time, actuator state and warmstart are
    /// NOT restored). Exists to demonstrate why the full integration state is needed.
    pub fn restore_qpos_qvel_only(&mut self, qpos: &[f64], qvel: &[f64]) -> Result<(), PortError> {
        check_finite("qpos", qpos, self.dims.nq)?;
        check_finite("qvel", qvel, self.dims.nv)?;
        self.call(&format!(
            "{{\"cmd\":\"restore_partial\",\"qpos\":{},\"qvel\":{}}}",
            fmt_arr(qpos),
            fmt_arr(qvel)
        ))
        .map(|_| ())
    }

    /// Actuator state (`act`).
    pub fn act(&mut self) -> Result<Vec<f64>, PortError> {
        let r = self.get()?;
        Self::vec_field(&r, "act")
    }
}

impl Drop for MujocoSidecarEngine {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl ContinuousPhysicsEnginePort for MujocoSidecarEngine {
    fn name(&self) -> &str {
        TAG
    }
    fn dims(&self) -> Dims {
        self.dims
    }
    fn dt(&self) -> f64 {
        self.dt
    }
    fn time(&mut self) -> Result<f64, PortError> {
        self.get()?
            .get("time")
            .and_then(Json::as_f64)
            .ok_or_else(|| PortError::Backend("missing time".into()))
    }
    fn qpos(&mut self) -> Result<Vec<f64>, PortError> {
        let r = self.get()?;
        Self::vec_field(&r, "qpos")
    }
    fn qvel(&mut self) -> Result<Vec<f64>, PortError> {
        let r = self.get()?;
        Self::vec_field(&r, "qvel")
    }
    fn ctrl(&mut self) -> Result<Vec<f64>, PortError> {
        let r = self.get()?;
        Self::vec_field(&r, "ctrl")
    }
    fn body_twist(&mut self) -> Result<(f64, f64), PortError> {
        let r = self.get()?;
        let g = |k: &str| {
            r.get(k)
                .and_then(Json::as_f64)
                .ok_or_else(|| PortError::Backend(format!("missing {k}")))
        };
        Ok((g("v")?, g("w")?))
    }
    fn set_ctrl(&mut self, ctrl: &[f64]) -> Result<(), PortError> {
        check_finite("ctrl", ctrl, self.dims.nu)?;
        self.call(&format!(
            "{{\"cmd\":\"set_ctrl\",\"ctrl\":{}}}",
            fmt_arr(ctrl)
        ))
        .map(|_| ())
    }
    fn step(&mut self, n: usize) -> Result<(), PortError> {
        self.call(&format!("{{\"cmd\":\"step\",\"n\":{n}}}"))
            .map(|_| ())
    }
    fn snapshot(&mut self) -> Result<EngineSnapshot, PortError> {
        let r = self.call("{\"cmd\":\"snapshot\"}")?;
        let state = Self::vec_field(&r, "state")?;
        let time = r.get("time").and_then(Json::as_f64).unwrap_or(0.0);
        let ctrl = self.ctrl()?;
        Ok(EngineSnapshot {
            backend: TAG.into(),
            time,
            ctrl,
            state,
        })
    }
    fn restore(&mut self, s: &EngineSnapshot) -> Result<(), PortError> {
        if s.backend != TAG {
            return Err(PortError::SnapshotMismatch(format!(
                "snapshot from {}",
                s.backend
            )));
        }
        check_finite("state", &s.state, s.state.len())?;
        self.call(&format!(
            "{{\"cmd\":\"restore\",\"state\":{}}}",
            fmt_arr(&s.state)
        ))
        .map(|_| ())
    }
    fn reset(&mut self) -> Result<(), PortError> {
        self.call("{\"cmd\":\"reset\"}").map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn cfg() -> SidecarConfig {
        SidecarConfig {
            call_timeout: Duration::from_millis(400),
            init_timeout: Duration::from_millis(400),
            shutdown_grace: Duration::from_millis(300),
        }
    }

    fn fake(script: &str) -> MujocoSidecarEngine {
        let mut c = Command::new("python3");
        c.args(["-P", "-B", "-c", script]);
        MujocoSidecarEngine::from_command(c, cfg()).expect("spawn fake")
    }

    fn reaped(pid: u32) -> bool {
        !Path::new(&format!("/proc/{pid}")).exists()
    }

    const HANG: &str = "import sys,time\nsys.stdin.readline()\ntime.sleep(60)";
    const HANG_IGNORE_TERM: &str = "import signal,time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nwhile True: time.sleep(1)";
    const EOF: &str =
        "import sys\nsys.stdin.readline()\nimport os\nos.close(1)\nimport time\ntime.sleep(60)";
    const GARBAGE: &str = "import sys\nsys.stdin.readline()\nprint('not json',flush=True)\nimport time\ntime.sleep(60)";
    const ERR: &str = "import sys\nsys.stdin.readline()\nprint('{\"ok\":false,\"error\":\"boom\"}',flush=True)\nsys.stdin.readline()";

    #[test]
    fn gog_142_hang_times_out_typed_and_reaps() {
        let mut e = fake(HANG);
        let pid = e.pid();
        let t = Instant::now();
        assert!(matches!(e.call("{}"), Err(PortError::Timeout(_))));
        assert!(t.elapsed() < Duration::from_secs(2));
        assert!(reaped(pid), "child must be killed and reaped on timeout");
        assert!(matches!(e.call("{}"), Err(PortError::Closed(_))));
    }

    #[test]
    fn gog_142_drop_kills_unresponsive_child_bounded() {
        let mut c = Command::new("python3");
        c.args(["-P", "-B", "-c", HANG_IGNORE_TERM]);
        let e = MujocoSidecarEngine::from_command(c, cfg()).unwrap();
        let pid = e.pid();
        let t = Instant::now();
        drop(e);
        assert!(t.elapsed() < Duration::from_secs(2));
        assert!(reaped(pid));
    }

    #[test]
    fn gog_142_eof_is_typed_and_closes() {
        let mut e = fake(EOF);
        let pid = e.pid();
        match e.call("{}") {
            Err(PortError::Backend(m)) => assert!(m.contains("closed stdout"), "{m}"),
            other => panic!("unexpected {other:?}"),
        }
        assert!(reaped(pid));
        assert!(matches!(e.call("{}"), Err(PortError::Closed(_))));
    }

    #[test]
    fn gog_142_invalid_json_closes() {
        let mut e = fake(GARBAGE);
        let pid = e.pid();
        assert!(matches!(e.call("{}"), Err(PortError::Backend(m)) if m.starts_with("json")));
        assert!(reaped(pid));
        assert!(matches!(e.call("{}"), Err(PortError::Closed(_))));
    }

    #[test]
    fn gog_142_error_reply_keeps_sidecar_usable_and_close_reaps() {
        let mut e = fake(ERR);
        let pid = e.pid();
        assert!(matches!(e.call("{}"), Err(PortError::Backend(m)) if m == "boom"));
        e.close();
        assert!(reaped(pid));
        e.close(); // idempotent
    }

    #[test]
    fn gog_142_ok_roundtrip_and_stderr_bounded() {
        let mut e = fake(
            "import sys\nsys.stderr.write('x'*100000)\nsys.stderr.flush()\nfor l in sys.stdin:\n    print('{\"ok\":true}',flush=True)",
        );
        assert!(e.call("{}").is_ok());
        assert!(e.call("{}").is_ok());
        assert!(e.stderr_tail().len() <= STDERR_CAP);
        let pid = e.pid();
        drop(e);
        assert!(reaped(pid));
    }

    #[test]
    fn gog_142_spawn_failure_is_typed() {
        let c = Command::new("/nonexistent/gog142");
        assert!(matches!(
            MujocoSidecarEngine::from_command(c, cfg()),
            Err(PortError::Backend(_))
        ));
    }
}
