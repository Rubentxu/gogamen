#!/usr/bin/env python3
"""CI helper (GOG-144). Stdlib only, deterministic, offline.

  ci_affected.py affected BASE HEAD   -> prints `ALL` or a space-separated package list (may be empty = nothing to test)
  ci_affected.py packages             -> every testable workspace package (EXCLUDED removed)
  ci_affected.py fmt-flags            -> `-p a -p b ...` for `cargo fmt --check` (ros2 excluded)
  ci_affected.py flags BASE HEAD      -> affected as `-p a -p b` flags, or `--workspace --exclude ...` when ALL
  ci_affected.py contract BASE HEAD   -> `RUN` if the canonical-JSON contract changed (GOG-152), else `SKIP`

Mapping: `git diff --name-only BASE...HEAD` -> owning workspace package by manifest directory -> plus transitive
reverse dependents (path deps). ALL (fallback) if Cargo.lock, root Cargo.toml, rust-toolchain.toml, deny.toml,
.cargo/, scripts/ or .github/ change, or if a changed file is outside every package and is not docs/markdown.
Excluded from test/clippy/fmt (needs a ROS2 toolchain / pre-existing rustfmt diffs; see docs/RELEASE.md "CI"):
"""
import json, os, subprocess, sys

EXCLUDED = {"gogamen-lab-ros2"}
ALL_TRIGGERS = ("Cargo.lock", "Cargo.toml", "rust-toolchain.toml", "rust-toolchain", "deny.toml")
ALL_PREFIXES = ("scripts/", ".github/", ".cargo/")
IGNORED_PREFIXES = ("docs/", "seed/", "python/", "kotlin/")  # no Rust impact (kotlin/python have own jobs)

# GOG-152: Rust and Kotlin canonical JSON serializers are compared through a shared golden corpus,
# only when one of these files changes (the contract), not on every PR.
CONTRACT_FILES = (
    "crates/gogamen-spec/src/codec.rs",
    "crates/gogamen-spec/src/json.rs",
    "crates/gogamen-spec/tests/gog_152_canonical_json_corpus.rs",
    "kotlin/src/main/kotlin/dev/gogamen/dsl/CanonicalJson.kt",
    "kotlin/src/test/kotlin/dev/gogamen/dsl/Gog152CanonicalJsonCorpusTest.kt",
)
CONTRACT_PREFIXES = ("spec/golden/canonical_json/",)

def contract_changed(base, head):
    files = subprocess.run(["git", "diff", "--name-only", f"{base}...{head}"], check=True,
                           capture_output=True, text=True).stdout.split()
    return any(f in CONTRACT_FILES or f.startswith(CONTRACT_PREFIXES) for f in files)

def meta():
    out = subprocess.run(["cargo", "metadata", "--offline", "--no-deps", "--format-version", "1"],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)

def pkgs(m):
    root = m["workspace_root"]
    return {p["name"]: (os.path.relpath(os.path.dirname(p["manifest_path"]), root), p) for p in m["packages"]}

def affected(base, head):
    files = subprocess.run(["git", "diff", "--name-only", f"{base}...{head}"], check=True,
                           capture_output=True, text=True).stdout.split()
    m = meta(); P = pkgs(m)
    hit = set()
    for f in files:
        if f in ALL_TRIGGERS or f.startswith(ALL_PREFIXES):
            return None
        owner = [n for n, (d, _) in P.items() if f.startswith(d + "/")]
        if owner:
            hit.update(owner)
        elif not (f.startswith(IGNORED_PREFIXES) or f.endswith(".md") or f in ("LICENSE", ".gitignore")
                  or "/" not in f and f.endswith(".py")):
            return None
    rdeps = {n: set() for n in P}
    for n, (_, p) in P.items():
        for d in p["dependencies"]:
            if d["name"] in rdeps:
                rdeps[d["name"]].add(n)
    todo = list(hit)
    while todo:
        for r in rdeps[todo.pop()]:
            if r not in hit:
                hit.add(r); todo.append(r)
    return sorted(hit - EXCLUDED)

def flags(names):
    return " ".join(f"-p {n}" for n in names)

def main(a):
    cmd = a[0] if a else ""
    if cmd == "affected" and len(a) == 3:
        r = affected(a[1], a[2]); print("ALL" if r is None else " ".join(r))
    elif cmd == "flags" and len(a) == 3:
        r = affected(a[1], a[2])
        print("--workspace " + " ".join(f"--exclude {e}" for e in sorted(EXCLUDED)) if r is None else flags(r))
    elif cmd == "contract" and len(a) == 3:
        print("RUN" if contract_changed(a[1], a[2]) else "SKIP")
    elif cmd == "packages":
        print(" ".join(sorted(set(pkgs(meta())) - EXCLUDED)))
    elif cmd == "fmt-flags":
        print(flags(sorted(set(pkgs(meta())) - EXCLUDED)))
    else:
        print(__doc__ + ", ".join(sorted(EXCLUDED)), file=sys.stderr); return 2
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
