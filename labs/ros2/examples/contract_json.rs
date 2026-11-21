//! Prints the ROS 2 contract as JSON (consumed by scripts/ros2_contract_smoke.sh).
fn main() {
    println!("{}", gogamen_lab_ros2::contract::contract_json());
}
