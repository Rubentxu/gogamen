//! `cargo run -p gogamen-lab-mario --example mario_run -- [--rom PATH] [--known FILE] [--fixture] [--seed N]`
//! Exit codes: 0 ok, 2 typed error, 3 NOT_RUN.
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut out = String::new();
    let code = gogamen_lab_mario::command::run(&args, &mut out);
    print!("{out}");
    std::process::exit(code);
}
