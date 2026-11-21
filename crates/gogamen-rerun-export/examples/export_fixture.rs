//! Exporta un fixture a un `.rrd` de Rerun.
//!
//! ```text
//! cargo run -p gogamen-rerun-export --features rerun --example export_fixture -- \
//!     <salida.rrd> [grid|scientific] [--allow-messages] [--allow-secrets]
//! rerun <salida.rrd>
//! ```
//! Sin `--allow-*` los mensajes y secretos se retienen/redactan (denegado por defecto).

use std::path::PathBuf;

use gogamen_rerun_export::fixtures::{grid_recording_with_gap, scientific_archive};
use gogamen_rerun_export::rerun_sink::{export_archive, export_recording};
use gogamen_rerun_export::{ExportOptions, ExportPermissions};

fn main() {
    let mut out = None;
    let mut source = "grid".to_owned();
    let mut permissions = ExportPermissions::default();
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--allow-messages" => permissions.messages = true,
            "--allow-secrets" => permissions.secrets = true,
            "grid" | "scientific" => source = arg,
            _ => out = Some(PathBuf::from(arg)),
        }
    }
    let Some(out) = out else {
        eprintln!(
            "usage: export_fixture <out.rrd> [grid|scientific] [--allow-messages] [--allow-secrets]"
        );
        std::process::exit(2);
    };
    let options = ExportOptions {
        permissions,
        ..ExportOptions::default()
    };
    let result = if source == "grid" {
        export_recording(&grid_recording_with_gap(40, 10, 14), &options, &out)
    } else {
        export_archive(&scientific_archive(), &options, &out)
    };
    match result {
        Ok(()) => println!("wrote {}", out.display()),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
