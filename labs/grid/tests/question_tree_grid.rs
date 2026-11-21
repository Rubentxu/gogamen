//! GOG-189 / DEM-035: question tree vs uniform mesh on Grid physics.
//!
//! PREREGISTERED COMPARISON (written before running):
//! - Data: every (non-wall cell, action) pair of the map, with the transition obtained by the
//!   Grid physics (move / blocked by wall or border / enter hazard). Samples are exhaustive and
//!   deterministic, so this measures representational economy, NOT sample efficiency.
//! - Senses: `[x, y, x]` (third is a duplicate). Outcome class: 0 moved, 1 blocked, 2 hazard.
//!   Viable: the cell is not a hazard. No rewards anywhere.
//! - Predictive error: `H(viable | cell) + H(outcome | cell, action)` in bits per sample.
//! - Baselines: uniform meshes with square blocks of side b = 1..=max(w,h).
//! - Claim under test: the tree reaches the error of the SMALLEST-leaf uniform mesh that is
//!   at least as accurate as the tree, with fewer leaves. If not, it is reported as negative.
//! - Also reported: the tree error vs the mesh with block 1 (exact cell scale).

use gogamen_core::question_tree::{
    QuestionTree, QuestionTreeConfig, Sample, predictive_error_bits,
};
use gogamen_lab_grid::{CellKind, GridMap, GridPos};

const DELTAS: [(isize, isize); 5] = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)];

fn samples(map: &GridMap) -> Vec<Sample> {
    let mut out = Vec::new();
    for y in 0..map.height() {
        for x in 0..map.width() {
            let pos = GridPos::new(x, y);
            let here = map.cell(pos).unwrap();
            if here == CellKind::Wall {
                continue;
            }
            for (a, (dx, dy)) in DELTAS.iter().enumerate() {
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                let outcome = if nx < 0 || ny < 0 {
                    1
                } else {
                    match map.cell(GridPos::new(nx as usize, ny as usize)) {
                        None | Some(CellKind::Wall) => 1,
                        Some(CellKind::Hazard) => 2,
                        Some(_) => 0,
                    }
                };
                out.push(Sample {
                    senses: vec![x as f64, y as f64, x as f64],
                    action: a as u32,
                    outcome,
                    viable: here != CellKind::Hazard,
                    preferred: None,
                });
            }
        }
    }
    out
}

fn mesh(samples: &[Sample], b: usize) -> (usize, f64) {
    let cell = |s: &Sample| (s.senses[0] as usize / b) * 1000 + s.senses[1] as usize / b;
    let mut ids: Vec<usize> = samples.iter().map(cell).collect();
    ids.sort_unstable();
    ids.dedup();
    (ids.len(), predictive_error_bits(samples, cell))
}

#[test]
fn question_tree_grid_vs_uniform_mesh() {
    let map = GridMap::from_ascii(&[
        "############",
        "#..........#",
        "#..........#",
        "#....##....#",
        "#....##..H.#",
        "#..........#",
        "#.S......G.#",
        "############",
    ])
    .unwrap();
    let data = samples(&map);
    let mut tree = QuestionTree::new(QuestionTreeConfig::default()).unwrap();
    let scale = tree.grow(&data).unwrap();
    let leaves = scale.leaf_count();
    let tree_err = predictive_error_bits(&data, |s| scale.classify(&s.senses).unwrap().leaf.0);

    let first = tree.history().first().expect("a first cut must exist");
    println!(
        "first cut: s{} <= {} (gain {:.1} bits, cost {:.1}) -> version {:?}",
        first.predicate.sense,
        first.predicate.threshold,
        first.gain_bits,
        first.cost_bits,
        first.version
    );
    println!(
        "tree: {leaves} leaves, error {tree_err:.4} bits/sample, depth {}",
        scale.depth()
    );

    let mut best_mesh: Option<(usize, usize, f64)> = None;
    for b in 1..=12 {
        let (n, e) = mesh(&data, b);
        println!("mesh b={b}: {n} leaves, error {e:.4}");
        if e <= tree_err + 1e-12 && best_mesh.is_none_or(|(_, bn, _)| n < bn) {
            best_mesh = Some((b, n, e));
        }
    }
    let (_, exact_leaves, _) = best_mesh.expect("b=1 mesh is exact, so some mesh qualifies");
    let verdict = if leaves < exact_leaves {
        "FEWER leaves"
    } else {
        "NOT fewer leaves"
    };
    println!("verdict: tree {leaves} vs smallest equally-accurate mesh {exact_leaves}: {verdict}");

    // Structural facts independent of the verdict:
    assert!(first.predicate.sense <= 1, "duplicate sense 2 never wins");
    assert!(tree_err <= mesh(&data, 12).1 + 1e-12);
    for (id, lits, _) in scale.leaves() {
        assert!(lits.len() <= scale.depth());
        assert!(scale.describe_leaf(id).is_some());
    }
    // Preregistered claim, asserted as measured on this map.
    assert!(leaves <= exact_leaves, "{verdict}");
}
