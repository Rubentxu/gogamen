//! Self-contained HTML report for an [`ExperimentOutcome`].
//!
//! One static file: inline CSS, inline SVG, no JavaScript, no external resources, no network.

use std::fmt::Write as _;

use gogamen_arena::HypothesisVerdict;

use crate::receipt::{ExperimentOutcome, RoleCurve};

const COLORS: [&str; 5] = ["#1b7f3b", "#2563eb", "#b45309", "#7c3aed", "#475569"];

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => o.push_str("&amp;"),
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            '"' => o.push_str("&quot;"),
            '\'' => o.push_str("&#39;"),
            c => o.push(c),
        }
    }
    o
}

fn curves_svg(curves: &[RoleCurve], ticks: usize, y_max: f64) -> String {
    let (w, h, l, r, t, b) = (
        760.0_f64, 340.0_f64, 48.0_f64, 150.0_f64, 12.0_f64, 34.0_f64,
    );
    let (pw, ph) = (w - l - r, h - t - b);
    let y_max = y_max.max(1.0);
    let x = |i: usize| l + pw * i as f64 / (ticks.max(2) - 1) as f64;
    let y = |v: f64| t + ph * (1.0 - (v / y_max).clamp(0.0, 1.0));
    let mut s = format!(
        "<svg viewBox=\"0 0 {w} {h}\" width=\"{w}\" height=\"{h}\" role=\"img\" aria-label=\"Checkpoints cleared per tick, mean with one standard deviation band per role\">\n"
    );
    let _ = writeln!(
        s,
        "<rect x=\"{l}\" y=\"{t}\" width=\"{pw}\" height=\"{ph}\" fill=\"#fff\" stroke=\"#999\"/>"
    );
    for k in 0..=4 {
        let v = y_max * f64::from(k) / 4.0;
        let _ = writeln!(
            s,
            "<line x1=\"{l}\" x2=\"{}\" y1=\"{yy:.1}\" y2=\"{yy:.1}\" stroke=\"#eee\"/><text x=\"{}\" y=\"{:.1}\" font-size=\"11\" text-anchor=\"end\">{v:.1}</text>",
            l + pw,
            l - 4.0,
            y(v) + 4.0,
            yy = y(v)
        );
    }
    let _ = writeln!(
        s,
        "<text x=\"{}\" y=\"{}\" font-size=\"11\" text-anchor=\"middle\">tick (0..{})</text>",
        l + pw / 2.0,
        h - 8.0,
        ticks.saturating_sub(1)
    );
    for (idx, c) in curves.iter().enumerate() {
        let color = COLORS[idx % COLORS.len()];
        let dash = if c.privileged {
            " stroke-dasharray=\"6 3\""
        } else {
            ""
        };
        // Band: mean +/- std, clamped to the observed [min, max] of the seeds.
        let mut pts = String::new();
        for i in 0..ticks {
            let hi = (c.mean[i] + c.std_dev[i]).min(c.max[i]);
            let _ = write!(pts, "{:.1},{:.1} ", x(i), y(hi));
        }
        for i in (0..ticks).rev() {
            let lo = (c.mean[i] - c.std_dev[i]).max(c.min[i]);
            let _ = write!(pts, "{:.1},{:.1} ", x(i), y(lo));
        }
        let _ = writeln!(
            s,
            "<polygon points=\"{}\" fill=\"{color}\" fill-opacity=\"0.18\" stroke=\"none\"/>",
            pts.trim_end()
        );
        let mut line = String::new();
        for i in 0..ticks {
            let _ = write!(line, "{:.1},{:.1} ", x(i), y(c.mean[i]));
        }
        let _ = writeln!(
            s,
            "<polyline points=\"{}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2\"{dash}/>",
            line.trim_end()
        );
        let ly = t + 14.0 + 18.0 * idx as f64;
        let _ = writeln!(
            s,
            "<line x1=\"{}\" x2=\"{}\" y1=\"{ly}\" y2=\"{ly}\" stroke=\"{color}\" stroke-width=\"2\"{dash}/><text x=\"{}\" y=\"{}\" font-size=\"11\">{}</text>",
            l + pw + 8.0,
            l + pw + 28.0,
            l + pw + 32.0,
            ly + 4.0,
            esc(&c.role)
        );
    }
    s.push_str("</svg>\n");
    s
}

/// Renders the full report.
pub fn render_html(o: &ExperimentOutcome, receipt_hash: u64) -> String {
    let mut h = String::new();
    h.push_str("<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\n");
    let _ = writeln!(
        h,
        "<title>gogamen experiment report - {}</title>",
        esc(&o.spec.example_id)
    );
    h.push_str("<style>body{font:14px/1.45 system-ui,sans-serif;max-width:980px;margin:2em auto;padding:0 1em;color:#111}table{border-collapse:collapse;margin:.6em 0}td,th{border:1px solid #bbb;padding:3px 8px;text-align:left;vertical-align:top}th{background:#f1f1f1}code{background:#f3f3f3;padding:0 3px}.b{padding:1px 8px;border-radius:9px;font-weight:600;color:#fff}.SUPPORTED{background:#1b7f3b}.REFUTED{background:#b91c1c}.INCONCLUSIVE{background:#6b7280}.note{background:#fff7e0;border-left:4px solid #e0a800;padding:.4em .8em}.priv{color:#7c3aed}</style>\n</head><body>\n");
    let _ = writeln!(h, "<h1>Experiment report: {}</h1>", esc(&o.spec.example_id));
    h.push_str("<p>Pure reward-free controller versus random, with the goal-oriented FMC and a privileged FMC with 4x the transition budget as a ceiling (reported separately, never part of the fair comparison).</p>\n");
    let _ = writeln!(
        h,
        "<h2>Receipt</h2><table><tr><th>gogamen version</th><td>{}</td></tr><tr><th>spec hash</th><td><code>{:016x}</code></td></tr><tr><th>pre-registered hypotheses hash</th><td><code>{}</code></td></tr><tr><th>report hash</th><td><code>{:016x}</code></td></tr><tr><th>receipt hash</th><td><code>{:016x}</code></td></tr><tr><th>track / ticks / horizon / transitions per decision</th><td>{} / {} / {} / {}</td></tr><tr><th>dev seeds</th><td>{:?}</td></tr><tr><th>holdout seeds</th><td>{:?}</td></tr></table>",
        esc(env!("CARGO_PKG_VERSION")),
        o.spec_hash,
        o.registered_hypotheses_hash.map_or_else(
            || "none (NOT pre-registered)".to_owned(),
            |v| format!("{v:016x}")
        ),
        o.report_hash,
        receipt_hash,
        esc(&o.spec.track),
        o.spec.max_ticks,
        o.spec.horizon,
        o.spec.transitions_per_decision,
        o.spec.dev_seeds,
        o.spec.holdout_seeds
    );
    h.push_str("<p>Verify by re-execution: <code>gogamen verify-receipt --receipt-file receipt.json</code></p>\n");

    h.push_str("<h2>Verdicts</h2>\n");
    if !o.preregistered {
        h.push_str("<p class=\"note\">The hypotheses were not pre-registered for this run, so none of them can be SUPPORTED.</p>\n");
    }
    h.push_str("<table><tr><th>hypothesis</th><th>claim</th><th>metric</th><th>observed</th><th>verdict</th><th>basis</th><th>build</th><th>evidence</th></tr>\n");
    let (mut sup, mut refu, mut inc) = (0, 0, 0);
    for v in &o.verdicts {
        match v.verdict {
            HypothesisVerdict::Supported => sup += 1,
            HypothesisVerdict::Refuted => refu += 1,
            HypothesisVerdict::Inconclusive => inc += 1,
        }
        let _ = writeln!(
            h,
            "<tr><td>{}</td><td>{}</td><td><code>{}</code></td><td>{}</td><td><span class=\"b {}\">{}</span></td><td>{}</td><td>{}</td><td><code>{}</code></td></tr>",
            esc(&v.hypothesis_id),
            esc(&v.claim),
            esc(&v.metric),
            v.observed
                .map_or_else(|| "n/a".to_owned(), |x| format!("{x:.4}")),
            v.verdict.label(),
            v.verdict.label(),
            v.basis.label(),
            v.build.label(),
            esc(&v.evidence_ref)
        );
    }
    let _ = writeln!(
        h,
        "</table><p>SUPPORTED={sup} REFUTED={refu} INCONCLUSIVE={inc}. INCONCLUSIVE is never counted as SUPPORTED. Verdict is independent of build status (NOT_RUN: this receipt does not run the code checks).</p>"
    );

    h.push_str("<h2>Endowment ledger</h2><table><tr><th>quantity</th><th>value</th><th>provenance</th><th>given to</th></tr>\n");
    for l in &o.ledger {
        let _ = writeln!(
            h,
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
            esc(&l.name),
            esc(&l.value),
            l.provenance.label(),
            esc(&l.scope)
        );
    }
    h.push_str("</table>\n");

    h.push_str("<h2>Summary per role (all seeds, nothing dropped)</h2><table><tr><th>role</th><th>episodes</th><th>succeeded</th><th>failed</th><th>timed out</th><th>efficacy mean</th><th>std</th><th>min</th><th>q25</th><th>median</th><th>q75</th><th>max</th></tr>\n");
    for s in &o.summaries {
        let d = &s.efficacy_dispersion;
        let cls = if s.is_privileged {
            " class=\"priv\""
        } else {
            ""
        };
        let _ = writeln!(
            h,
            "<tr{cls}><td>{}{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td><td>{:.3}</td></tr>",
            s.role.label(),
            if s.is_privileged {
                " (privileged ceiling)"
            } else {
                ""
            },
            s.total_runs,
            s.succeeded_count,
            s.failed_count,
            s.timed_out_count,
            d.mean,
            d.std_dev,
            d.min,
            d.q25,
            d.median,
            d.q75,
            d.max
        );
    }
    h.push_str(
        "</table><p>Efficacy = checkpoints cleared / checkpoints of the track, per seed.</p>\n",
    );

    h.push_str("<h2>Curves with dispersion</h2><p>Checkpoints cleared per tick; line = mean over seeds, band = mean +/- 1 std (clipped to the seeds' min and max). Dashed = privileged ceiling.</p>\n");
    let y_max = o
        .curves
        .iter()
        .flat_map(|c| c.max.iter().copied())
        .fold(1.0_f64, f64::max);
    h.push_str(&curves_svg(&o.curves, o.spec.max_ticks, y_max));

    let failed: Vec<_> = o.runs.iter().filter(|r| r.reason.is_some()).collect();
    let _ = writeln!(
        h,
        "<h2>Failed episodes</h2><p>{} of {} episodes did not reach the goal; all are listed.</p>",
        failed.len(),
        o.runs.len()
    );
    h.push_str("<table><tr><th>role</th><th>split</th><th>seed</th><th>status</th><th>reason</th><th>ticks</th><th>checkpoints</th><th>control hash</th></tr>\n");
    for r in failed {
        let _ = writeln!(
            h,
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td><code>{:016x}</code></td></tr>",
            esc(&r.role),
            esc(&r.split),
            r.seed,
            esc(&r.status),
            esc(r.reason.as_deref().unwrap_or("")),
            r.steps,
            r.checkpoints,
            r.control_hash
        );
    }
    h.push_str("</table>\n<h2>Limits</h2><p>Fixed track, fixed seeds, short episodes: the numbers describe these executions only. 64-bit FNV-1a hashes detect edits and non-reproducible runs; they are not signatures.</p>\n</body></html>\n");
    h
}
