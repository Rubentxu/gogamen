//! Typed ROS 2 contract between the (remote, best-effort) planner and the robot-side controller.
//!
//! Message structs mirror `geometry_msgs/msg/Twist`, `nav_msgs/msg/Odometry` (reduced to planar
//! fields) and `sensor_msgs/msg/LaserScan` (reduced). Times are virtual milliseconds.

/// 3-vector, same shape as `geometry_msgs/msg/Vector3` (all `float64`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// `geometry_msgs/msg/Twist`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Twist {
    pub linear: Vec3,
    pub angular: Vec3,
}

impl Twist {
    /// Planar command: forward speed (m/s) and yaw rate (rad/s).
    pub fn planar(v: f64, w: f64) -> Self {
        Self { linear: Vec3 { x: v, y: 0.0, z: 0.0 }, angular: Vec3 { x: 0.0, y: 0.0, z: w } }
    }
    pub const ZERO: Twist = Twist {
        linear: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
        angular: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
    };
    pub fn is_finite(&self) -> bool {
        [self.linear.x, self.linear.y, self.linear.z, self.angular.x, self.angular.y, self.angular.z]
            .iter()
            .all(|v| v.is_finite())
    }
    pub fn is_zero(&self) -> bool {
        *self == Twist::ZERO
    }
}

/// Reduced planar `nav_msgs/msg/Odometry`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Odom {
    pub stamp_ms: u64,
    pub x: f64,
    pub y: f64,
    pub yaw: f64,
    pub v: f64,
    pub w: f64,
}

/// Reduced `sensor_msgs/msg/LaserScan`.
#[derive(Debug, Clone, PartialEq)]
pub struct Scan {
    pub stamp_ms: u64,
    pub angle_min: f32,
    pub angle_increment: f32,
    pub range_min: f32,
    pub range_max: f32,
    pub ranges: Vec<f32>,
}

/// Planner liveness state carried by the heartbeat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlannerState {
    Planning,
    Idle,
    Degraded,
}

/// `gogamen_msgs/msg/Heartbeat` (custom message; not part of stock ROS 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Heartbeat {
    pub stamp_ms: u64,
    pub seq: u64,
    pub state: PlannerState,
}

/// Robot-side mode reported on `/gogamen/robot_state`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Run,
    Stop,
    EStop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    PlannerToRobot,
    RobotToPlanner,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reliability {
    Reliable,
    BestEffort,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Durability {
    Volatile,
    TransientLocal,
}

/// QoS profile (history is always KEEP_LAST).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qos {
    pub reliability: Reliability,
    pub durability: Durability,
    pub depth: u32,
    /// DEADLINE policy period in ms: the publisher must publish at least this often.
    pub deadline_ms: u64,
}

/// One topic of the contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopicSpec {
    pub name: &'static str,
    pub msg_type: &'static str,
    pub direction: Direction,
    pub qos: Qos,
    pub rate_hz: u32,
}

const fn qos(r: Reliability, d: Durability, depth: u32, deadline_ms: u64) -> Qos {
    Qos { reliability: r, durability: d, depth, deadline_ms }
}

/// The contract. Command topics are depth 1 best-effort: stale commands are worthless, so they are
/// never queued or retransmitted; safety state is reliable and transient-local.
pub const TOPICS: [TopicSpec; 5] = [
    TopicSpec { name: "/cmd_vel", msg_type: "geometry_msgs/msg/Twist", direction: Direction::PlannerToRobot,
        qos: qos(Reliability::BestEffort, Durability::Volatile, 1, 100), rate_hz: 10 },
    TopicSpec { name: "/gogamen/heartbeat", msg_type: "gogamen_msgs/msg/Heartbeat", direction: Direction::PlannerToRobot,
        qos: qos(Reliability::BestEffort, Durability::Volatile, 1, 200), rate_hz: 5 },
    TopicSpec { name: "/odom", msg_type: "nav_msgs/msg/Odometry", direction: Direction::RobotToPlanner,
        qos: qos(Reliability::BestEffort, Durability::Volatile, 5, 100), rate_hz: 20 },
    TopicSpec { name: "/scan", msg_type: "sensor_msgs/msg/LaserScan", direction: Direction::RobotToPlanner,
        qos: qos(Reliability::BestEffort, Durability::Volatile, 5, 200), rate_hz: 10 },
    TopicSpec { name: "/gogamen/robot_state", msg_type: "gogamen_msgs/msg/RobotState", direction: Direction::RobotToPlanner,
        qos: qos(Reliability::Reliable, Durability::TransientLocal, 1, 500), rate_hz: 5 },
];

/// Field layout of `geometry_msgs/msg/Twist` as `(path, ros_type)`, in `ros2 interface show` order.
pub const TWIST_FIELDS: [(&str, &str); 6] = [
    ("linear.x", "float64"),
    ("linear.y", "float64"),
    ("linear.z", "float64"),
    ("angular.x", "float64"),
    ("angular.y", "float64"),
    ("angular.z", "float64"),
];

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Contract as JSON (hand-built; the crate has no serde). Used by the optional container smoke.
pub fn contract_json() -> String {
    let topics: Vec<String> = TOPICS
        .iter()
        .map(|t| {
            format!(
                "{{\"name\":\"{}\",\"type\":\"{}\",\"direction\":\"{}\",\"reliability\":\"{}\",\"durability\":\"{}\",\"depth\":{},\"deadline_ms\":{},\"rate_hz\":{}}}",
                esc(t.name),
                esc(t.msg_type),
                match t.direction { Direction::PlannerToRobot => "planner_to_robot", Direction::RobotToPlanner => "robot_to_planner" },
                match t.qos.reliability { Reliability::Reliable => "reliable", Reliability::BestEffort => "best_effort" },
                match t.qos.durability { Durability::Volatile => "volatile", Durability::TransientLocal => "transient_local" },
                t.qos.depth,
                t.qos.deadline_ms,
                t.rate_hz
            )
        })
        .collect();
    let fields: Vec<String> =
        TWIST_FIELDS.iter().map(|(p, t)| format!("{{\"path\":\"{p}\",\"type\":\"{t}\"}}")).collect();
    format!(
        "{{\"simulation_only\":true,\"topics\":[{}],\"twist_fields\":[{}]}}",
        topics.join(","),
        fields.join(",")
    )
}
