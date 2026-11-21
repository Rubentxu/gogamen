//! Hardware-in-the-loop gate. There is NO physical backend in this crate: every request resolves
//! to a typed `NotRun`/`NotAuthorized` result and nothing physical is ever executed.

/// No code path in this crate can enable hardware.
pub const HARDWARE_ENABLED: bool = false;

/// Items that must all be true, and separately authorised, before any physical test may exist.
pub const PHYSICAL_TEST_CHECKLIST: &[&str] = &[
    "Written authorization from the project owner naming the device and the test date",
    "Identified low-risk device (small, speed-limited, low mass) with documented serial/ID",
    "Independent hardware e-stop (cuts drive power without software) tested before each session",
    "Robot lifted/wheels off ground for the first run; then a fenced area with margin beyond the geofence",
    "Local controller limits (speed, geofence, timeouts) verified on the device, not only in sim",
    "Trained operator within reach of the e-stop; second person observing",
    "Link-loss and heartbeat-loss behaviour demonstrated in the emulator with the same config",
    "Abort criteria and incident log defined in advance",
    "A physical backend implemented and reviewed in a separate task (none exists here)",
];

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HilRequest {
    /// Free-form reference to the written authorization, if any.
    pub authorization: Option<String>,
    pub device_id: Option<String>,
    /// Checklist items the requester claims are satisfied.
    pub checklist_ack: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotRunReason {
    NoAuthorization,
    NoDevice,
    ChecklistIncomplete,
    /// Everything was provided, but this crate has no physical backend.
    NoPhysicalBackend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HilOutcome {
    NotRun(NotRunReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HilError {
    NotAuthorized(NotRunReason),
}

/// Request a physical run. Always `Err(NotAuthorized)`; never touches hardware.
pub fn run_physical(req: &HilRequest) -> Result<HilOutcome, HilError> {
    let reason = if req.authorization.as_deref().is_none_or(str::is_empty) {
        NotRunReason::NoAuthorization
    } else if req.device_id.as_deref().is_none_or(str::is_empty) {
        NotRunReason::NoDevice
    } else if !PHYSICAL_TEST_CHECKLIST.iter().all(|i| req.checklist_ack.iter().any(|a| a == i)) {
        NotRunReason::ChecklistIncomplete
    } else {
        NotRunReason::NoPhysicalBackend
    };
    Err(HilError::NotAuthorized(reason))
}
