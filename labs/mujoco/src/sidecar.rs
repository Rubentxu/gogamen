//! MuJoCo backend as an optional Python sidecar (feature `mujoco-sidecar`).
//!
//! Spawns `python3 -P -B -c <embedded script>` (override the interpreter with the
//! `GOGAMEN_PYTHON` env var) and speaks JSON lines over stdin/stdout. The script uses
//! `mj_getState`/`mj_setState` with `mjSTATE_INTEGRATION`.
//!
//! Note: `python3 -I` is intentionally not used because isolated mode also disables the
//! per-user site-packages where `pip install --user mujoco` places the module.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

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

/// MuJoCo sidecar engine.
pub struct MujocoSidecarEngine {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    dims: Dims,
    dt: f64,
    version: String,
}

impl MujocoSidecarEngine {
    /// Spawn the sidecar and initialise the model.
    pub fn spawn() -> Result<Self, PortError> {
        let mut child = Command::new(python())
            .args(["-P", "-B", "-c", SCRIPT])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| PortError::Backend(format!("spawn failed: {e}")))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| PortError::Backend("no stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| PortError::Backend("no stdout".into()))?;
        let mut me = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            dims: Dims {
                nq: 0,
                nv: 0,
                nu: 0,
                na: 0,
            },
            dt: 0.0,
            version: String::new(),
        };
        let r = me.call("{\"cmd\":\"init\"}")?;
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
        Ok(me)
    }

    /// MuJoCo version reported by the sidecar.
    pub fn mujoco_version(&self) -> &str {
        &self.version
    }

    fn call(&mut self, line: &str) -> Result<Json, PortError> {
        writeln!(self.stdin, "{line}")
            .and_then(|()| self.stdin.flush())
            .map_err(|e| PortError::Backend(format!("write: {e}")))?;
        let mut buf = String::new();
        let n = self
            .stdout
            .read_line(&mut buf)
            .map_err(|e| PortError::Backend(format!("read: {e}")))?;
        if n == 0 {
            return Err(PortError::Backend("sidecar closed stdout".into()));
        }
        let v = json::parse(buf.trim()).map_err(|e| PortError::Backend(format!("json: {e}")))?;
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
        let _ = writeln!(self.stdin, "{{\"cmd\":\"quit\"}}");
        let _ = self.stdin.flush();
        let _ = self.child.wait();
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
