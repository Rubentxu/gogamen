//! Opt-in real smoke against a Jev endpoint using SYNTHETIC data only.
//!
//! Runs only when `GOGAMEN_JEV_URL` and `GOGAMEN_JEV_KEY` are set (optionally
//! `GOGAMEN_JEV_MODEL`, default `jev-default`). Otherwise prints `NOT_RUN` and exits 0 without
//! claiming a pass. The contract is a hypothesis: a FAIL here may mean the real service
//! differs from `gogamen-jev/1`, not that the adapter is wrong.
//!
//! `cargo run -p gogamen-jev --features http --example jev_smoke`

use gogamen_core::{OptionId, QueryId, ReplaySystemOneProvider, SystemMonotonicClock, TypedQuery};
use gogamen_jev::{EnvSecret, JevConfig, JevHttpProvider, QuerySource, UreqTransport};

fn main() {
    let (Ok(url), true) = (
        std::env::var("GOGAMEN_JEV_URL"),
        std::env::var("GOGAMEN_JEV_KEY").is_ok_and(|k| !k.trim().is_empty()),
    ) else {
        println!(
            "NOT_RUN: set GOGAMEN_JEV_URL and GOGAMEN_JEV_KEY to run the real Jev smoke (no PASS claimed)"
        );
        return;
    };
    let model = std::env::var("GOGAMEN_JEV_MODEL").unwrap_or_else(|_| "jev-default".into());
    let provider = JevHttpProvider::new(
        JevConfig::new(url, model),
        UreqTransport,
        Box::new(EnvSecret::new("GOGAMEN_JEV_KEY")),
        ReplaySystemOneProvider::default(),
    );
    // Synthetic, non-personal query: only typed identifiers are ever sent.
    let query = TypedQuery::choice(
        QueryId::new("smoke-synthetic-1").unwrap(),
        vec![
            OptionId::new("alpha").unwrap(),
            OptionId::new("beta").unwrap(),
        ],
        None,
    )
    .unwrap();
    let out = provider.query_detailed(&query, &SystemMonotonicClock::new());
    match (&out.source, &out.result) {
        (QuerySource::Live, Ok(r)) => println!("PASS(live): status={}", r.status()),
        (QuerySource::Fallback(cause), _) => println!("FAIL(degraded to replay): cause={cause}"),
        (QuerySource::Live, Err(_)) => println!("FAIL: live answer rejected by validation"),
    }
}
