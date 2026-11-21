"""MuJoCo sidecar for gogamen-lab-mujoco. JSON lines on stdin/stdout.

Uses mj_getState / mj_setState with mjSTATE_INTEGRATION (time, qpos, qvel, act,
warmstart, ctrl, applied forces, ...), i.e. the complete integration state.
"""
import json
import sys

import mujoco
import numpy as np

MJCF = """
<mujoco model="mobile_robot">
  <option timestep="0.01" integrator="Euler" gravity="0 0 0"/>
  <worldbody>
    <body name="chassis" pos="0 0 0">
      <inertial pos="0 0 0" mass="2" diaginertia="0.05 0.05 0.05"/>
      <joint name="x" type="slide" axis="1 0 0" damping="1"/>
      <joint name="y" type="slide" axis="0 1 0" damping="1"/>
      <joint name="yaw" type="hinge" axis="0 0 1" damping="0.05"/>
      <geom type="box" size="0.1 0.07 0.03" contype="0" conaffinity="0"/>
      <site name="drive" pos="0 0 0"/>
    </body>
  </worldbody>
  <actuator>
    <general name="thrust" site="drive" gear="1 0 0 0 0 0" dyntype="filter" dynprm="0.05"
             gaintype="fixed" gainprm="1" ctrllimited="true" ctrlrange="-5 5"/>
    <general name="torque" site="drive" gear="0 0 0 0 0 1" dyntype="filter" dynprm="0.05"
             gaintype="fixed" gainprm="1" ctrllimited="true" ctrlrange="-1 1"/>
    <!-- no-skid constraint approximation: stiff damping of lateral (site-frame y) velocity -->
    <general name="lateral_damper" site="drive" gear="0 1 0 0 0 0" gaintype="fixed" gainprm="0"
             biastype="affine" biasprm="0 0 -100"/>
  </actuator>
</mujoco>
"""

SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)
NU_USER = 2

m = mujoco.MjModel.from_xml_string(MJCF)
d = mujoco.MjData(m)
NSTATE = mujoco.mj_stateSize(m, SPEC)


def get_state():
    out = np.zeros(NSTATE)
    mujoco.mj_getState(m, d, out, SPEC)
    return out


def twist():
    yaw = d.qpos[2]
    v = d.qvel[0] * np.cos(yaw) + d.qvel[1] * np.sin(yaw)
    return float(v), float(d.qvel[2])


def reply(obj):
    sys.stdout.write(json.dumps(obj, allow_nan=False) + "\n")
    sys.stdout.flush()


def handle(req):
    cmd = req["cmd"]
    if cmd == "init":
        return {"nq": m.nq, "nv": m.nv, "nu": NU_USER, "na": m.na, "nstate": int(NSTATE),
                "dt": m.opt.timestep, "mujoco": mujoco.__version__, "spec": SPEC}
    if cmd == "get":
        v, w = twist()
        return {"time": d.time, "qpos": d.qpos.tolist(), "qvel": d.qvel.tolist(),
                "act": d.act.tolist(), "ctrl": d.ctrl[:NU_USER].tolist(), "v": v, "w": w}
    if cmd == "set_ctrl":
        c = req["ctrl"]
        if len(c) != NU_USER:
            raise ValueError("ctrl must have 2 entries")
        # ctrlrange clamping is applied by MuJoCo during stepping; clamp here too so that
        # the reported/snapshotted ctrl equals the effective one.
        lo = m.actuator_ctrlrange[:NU_USER, 0]
        hi = m.actuator_ctrlrange[:NU_USER, 1]
        d.ctrl[:NU_USER] = np.clip(np.array(c, dtype=float), lo, hi)
        return {}
    if cmd == "step":
        for _ in range(int(req["n"])):
            mujoco.mj_step(m, d)
        return {}
    if cmd == "snapshot":
        return {"state": get_state().tolist(), "time": d.time}
    if cmd == "restore":
        s = np.array(req["state"], dtype=float)
        if s.shape != (NSTATE,):
            raise ValueError("state has wrong size")
        mujoco.mj_setState(m, d, s, SPEC)
        mujoco.mj_forward(m, d)
        return {}
    if cmd == "restore_partial":
        # Deliberately INCOMPLETE restore (qpos/qvel only) for the negative test: time, act and
        # warmstart keep their current values.
        d.qpos[:] = req["qpos"]
        d.qvel[:] = req["qvel"]
        return {}
    if cmd == "reset":
        mujoco.mj_resetData(m, d)
        return {}
    raise ValueError("unknown cmd " + str(cmd))


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("cmd") == "quit":
                reply({"ok": True})
                return
            out = handle(req)
            out["ok"] = True
            reply(out)
        except Exception as e:  # protocol errors must not kill the loop silently
            reply({"ok": False, "error": repr(e)})


main()
