//! The scheduling core must stay pure, and this test is what makes that a rule
//! rather than an aspiration.
//!
//! Why it matters enough to police automatically: the term project compares
//! scheduling policies over synthetic workloads *and* runs them on real
//! hardware, and both must exercise the same policy objects. The moment a
//! policy can await, read a clock, or open a file, the offline harness either
//! cannot run it or has to substitute something else -- and then the published
//! comparison is measuring code that never ships.
//!
//! Failing this test is not necessarily a bug. It is a design decision that
//! needs making deliberately: either the data belongs in the snapshot, or the
//! logic belongs in admission control, which runs before the queue and may do
//! I/O.

use std::path::{Path, PathBuf};

/// Things a pure policy cannot do, and what to do instead.
const FORBIDDEN: &[(&str, &str)] = &[
    (
        "async fn",
        "policies are synchronous; put I/O in admission control",
    ),
    (
        ".await",
        "policies are synchronous; put I/O in admission control",
    ),
    (
        "SystemTime::now",
        "take `now` from the context instead, so the simulator controls time",
    ),
    (
        "Instant::now",
        "take `now` from the context instead, so the simulator controls time",
    ),
    ("std::fs", "a policy reads its inputs from the snapshot"),
    ("std::net", "a policy reads its inputs from the snapshot"),
    (
        "std::process",
        "a policy reads its inputs from the snapshot",
    ),
];

/// Crates that would drag an async runtime or a transport in with them.
const FORBIDDEN_DEPS: &[&str] = &["tokio", "reqwest", "rusqlite", "hyper"];

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("readable source dir") {
        let path = entry.expect("readable entry").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Strip comments so prose about `.await` does not trip the scan.
fn code_only(line: &str) -> &str {
    let trimmed = line.trim_start();
    if trimmed.starts_with("//") {
        return "";
    }
    match line.find("//") {
        Some(i) => &line[..i],
        None => line,
    }
}

#[test]
fn the_scheduling_core_does_no_io_and_never_reads_the_clock() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_sources(&src, &mut files);
    assert!(!files.is_empty(), "found no sources to check under {src:?}");

    let mut violations = Vec::new();
    for file in &files {
        let text = std::fs::read_to_string(file).expect("readable source");
        for (n, line) in text.lines().enumerate() {
            let code = code_only(line);
            for (needle, why) in FORBIDDEN {
                if code.contains(needle) {
                    violations.push(format!(
                        "{}:{}: `{needle}` -- {why}\n    {}",
                        file.display(),
                        n + 1,
                        line.trim()
                    ));
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "ferro-sched must stay pure and synchronous:\n{}",
        violations.join("\n")
    );
}

#[test]
fn the_scheduling_core_pulls_in_no_runtime() {
    let manifest =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
            .expect("readable manifest");

    // Only the dependency tables, so the explanatory comment above them -- which
    // names the very crates it forbids -- is not mistaken for a dependency.
    let deps: String = manifest
        .lines()
        .skip_while(|l| !l.trim_start().starts_with("[dependencies]"))
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");

    for dep in FORBIDDEN_DEPS {
        assert!(
            !deps.contains(dep),
            "ferro-sched must not depend on `{dep}`: the offline scheduler harness \
             links this crate, and a policy that needs a runtime cannot be replayed \
             deterministically"
        );
    }
}
