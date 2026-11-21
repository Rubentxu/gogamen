#!/usr/bin/env bash
# OPTIONAL smoke (GOG-81): checks that the Twist layout in the Rust contract JSON matches the real
# geometry_msgs/msg/Twist from a ros:jazzy container, and that a real topic round-trips.
# Not part of the default checks. Exit 0 = PASS, 2 = NOT_RUN (no container runtime / image), 1 = FAIL.
set -u
IMAGE="${ROS_IMAGE:-docker.io/library/ros:jazzy-ros-base}"
RT="$(command -v podman || command -v docker || true)"
[ -n "$RT" ] || { echo "NOT_RUN: no podman/docker"; exit 2; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
JSON="$(cargo run -q --offline -p gogamen-lab-ros2 --example contract_json)" || { echo "FAIL: contract_json"; exit 1; }
"$RT" image exists "$IMAGE" 2>/dev/null || timeout 300 "$RT" pull "$IMAGE" >/dev/null 2>&1 || { echo "NOT_RUN: cannot pull $IMAGE"; exit 2; }
NAME="gogamen-ros2-smoke-$$"
trap '"$RT" rm -f "$NAME" >/dev/null 2>&1' EXIT
CONTRACT="$JSON" timeout 180 "$RT" run --rm --name "$NAME" -e CONTRACT "$IMAGE" bash -c '
source /opt/ros/jazzy/setup.bash
ros2 interface show geometry_msgs/msg/Twist > /tmp/twist.txt || exit 1
python3 - <<PY || exit 1
import json, os, re, sys
c = json.loads(os.environ["CONTRACT"])
fields, parent = [], None
for line in open("/tmp/twist.txt"):
    if line.lstrip().startswith("#") or not line.strip(): continue
    m = re.match(r"^(\S+)\s+(\w+)\s*$", line.rstrip())
    if m and not line.startswith("\t"): parent = m.group(2); continue
    m = re.match(r"^\t(\S+)\s+(\w+)\s*$", line.rstrip())
    if m: fields.append((parent + "." + m.group(2), m.group(1)))
want = [(f["path"], f["type"]) for f in c["twist_fields"]]
if fields != want: print("MISMATCH", fields, want); sys.exit(1)
print("twist layout matches:", fields)
PY
(timeout 20 ros2 topic echo --once /cmd_vel geometry_msgs/msg/Twist > /tmp/echo.txt 2>&1 &)
sleep 4
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}, angular: {z: 0.1}}" >/dev/null 2>&1
sleep 3
grep -q "x: 0.2" /tmp/echo.txt && echo "topic round-trip OK" || { echo "topic round-trip FAILED"; cat /tmp/echo.txt; exit 1; }
'
rc=$?
[ $rc -eq 0 ] && echo "PASS" || echo "FAIL (rc=$rc)"
exit $([ $rc -eq 0 ] && echo 0 || echo 1)
