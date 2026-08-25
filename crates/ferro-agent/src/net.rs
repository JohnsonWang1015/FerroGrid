//! Point-to-point TCP throughput between two agents.
//!
//! `p2p_probe.py` already proves NCCL can connect; what nobody could see was
//! how fast. On this cluster that is the number that decides whether a job may
//! cross the network at all, and a node that negotiated 100 Mb/s looks exactly
//! like a healthy one until somebody measures it.
//!
//! Deliberately plain TCP rather than an NCCL benchmark: it needs no GPUs, no
//! image and no rendezvous, so it also works on a node whose CUDA is broken --
//! which is exactly when you want to know whether the network is at fault.

use anyhow::{Context, Result};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// 4 MiB per write: big enough that the syscall overhead disappears, small
/// enough not to sit in the kernel buffer for long at the end of the run.
const CHUNK: usize = 4 << 20;

/// Bind a listener that drains one connection, and return its port.
///
/// It accepts exactly one peer and goes away on its own: the alternative is a
/// permanently open port on every GPU node that reads whatever it is given.
pub async fn sink(seconds: u32) -> Result<u16> {
    let listener = TcpListener::bind(("0.0.0.0", 0)).await.context("bind sink")?;
    let port = listener.local_addr().context("sink addr")?.port();

    tokio::spawn(async move {
        // Long enough for the sender to connect and finish, then gone whatever
        // happens -- a probe that dies mid-run must not leave this behind.
        let budget = Duration::from_secs(seconds as u64 + 10);
        let _ = tokio::time::timeout(budget, async move {
            let Ok((mut stream, _)) = listener.accept().await else { return };
            let mut buf = vec![0u8; CHUNK];
            while let Ok(n) = stream.read(&mut buf).await {
                if n == 0 {
                    break;
                }
            }
        })
        .await;
    });

    Ok(port)
}

/// Send for `seconds` and report what actually went through.
pub async fn probe(host: &str, port: u16, seconds: u32) -> Result<(u64, f64)> {
    let mut stream = TcpStream::connect((host, port))
        .await
        .with_context(|| format!("connect {host}:{port}"))?;
    stream.set_nodelay(true).ok();

    let buf = vec![0u8; CHUNK];
    // One chunk before the clock starts: TCP slow start would otherwise be
    // charged to the link, and on a short run that is a real dent.
    stream.write_all(&buf).await.context("warmup write")?;

    let window = Duration::from_secs(seconds.max(1) as u64);
    let start = Instant::now();
    let mut sent = 0u64;
    while start.elapsed() < window {
        stream.write_all(&buf).await.context("write")?;
        sent += buf.len() as u64;
    }
    stream.shutdown().await.ok();
    Ok((sent, start.elapsed().as_secs_f64()))
}

/// Megabits per second, the unit link speeds are quoted in.
pub fn mbps(bytes: u64, seconds: f64) -> f64 {
    if seconds <= 0.0 {
        return 0.0;
    }
    (bytes as f64 * 8.0) / seconds / 1e6
}

/// What the interface holding `ip` negotiated, in Mb/s. `None` when the link
/// is virtual, down, or the kernel will not say.
pub fn link_speed_mbps(iface: &str) -> Option<u32> {
    let raw = std::fs::read_to_string(format!("/sys/class/net/{iface}/speed")).ok()?;
    // Virtual and down interfaces report -1 here rather than failing.
    raw.trim().parse::<i64>().ok().filter(|v| *v > 0).map(|v| v as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throughput_is_in_megabits() {
        // 125 MB in one second is a saturated gigabit link.
        assert!((mbps(125_000_000, 1.0) - 1000.0).abs() < 0.001);
        assert_eq!(mbps(1, 0.0), 0.0);
    }

    #[tokio::test]
    async fn a_probe_measures_the_loopback() {
        let port = sink(1).await.unwrap();
        let (bytes, secs) = probe("127.0.0.1", port, 1).await.unwrap();
        assert!(bytes > 0 && secs > 0.0);
        // Loopback is not a link, but it proves the two halves talk.
        assert!(mbps(bytes, secs) > 100.0);
    }
}
