//! Shared process management: timeout enforcement and D-state detection.
//!
//! Both `npu_runner.rs` (CLI npu-runner tool) and `native_hw.rs` (compiled
//! test.exe) need to spawn child processes that talk to NPU hardware via XRT.
//! If the kernel driver enters D-state (uninterruptible sleep from a stuck
//! ioctl), the child becomes unkillable and the test runner hangs forever.
//!
//! This module provides:
//! - `configure_process_group()`: puts the child in its own session via
//!   `setsid()` so `killpg()` can kill the entire child tree.
//! - `wait_with_timeout()`: polls the child, kills on timeout, detects D-state
//!   if the process survives SIGKILL.
//!
//! # D-state Detection
//!
//! After SIGKILL, a healthy process dies within milliseconds. If it's still
//! alive after 500ms, it's in D-state (stuck in a kernel code path that
//! cannot process signals). We confirm by reading `/proc/<pid>/status`.
//! The caller can then stop all hardware execution immediately rather than
//! spawning more processes into a jammed device.

use std::io::Read;
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

/// Extract exit code from ExitStatus, preserving signal information.
///
/// Normal exit: returns the exit code (0-255).
/// Signal death: returns -(signal number), e.g. -11 for SIGSEGV, -6 for SIGABRT.
/// This lets callers distinguish between a normal error exit and a crash.
fn exit_code_from_status(status: &ExitStatus) -> i32 {
    if let Some(code) = status.code() {
        return code;
    }
    // On Unix, signal() gives the signal that killed the process.
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = status.signal() {
            return -sig;
        }
    }
    -1 // shouldn't happen, but safe fallback
}

/// Outcome of a managed child process execution.
pub enum ProcessOutcome {
    /// Process exited normally (success or failure).
    Completed {
        stdout: String,
        stderr: String,
        exit_code: i32,
    },
    /// Process exceeded the timeout and was killed successfully.
    Timeout {
        stdout: String,
        stderr: String,
    },
    /// Process survived SIGKILL -- stuck in D-state (uninterruptible sleep).
    /// The device is wedged and no further hardware tests should run.
    Wedged {
        pid: u32,
        stdout: String,
        stderr: String,
    },
    /// Failed to spawn the child process at all.
    SpawnError(String),
}

/// How often to poll `try_wait()` while waiting for the child.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// How long to wait after SIGKILL before declaring D-state.
/// SIGKILL takes effect in <1ms on healthy systems. 500ms is very generous.
const REAP_GRACE: Duration = Duration::from_millis(500);

/// Configure a command to spawn its child in a new session (process group).
///
/// Uses POSIX `setsid()` via `pre_exec` so the child becomes a session
/// leader. This allows `killpg(pid, SIGKILL)` to kill the entire child tree
/// (including any XRT internal processes) rather than just the top-level PID.
///
/// # Safety
///
/// `pre_exec` runs between `fork()` and `exec()` in an async-signal-safe
/// context. `setsid()` is async-signal-safe per POSIX.
pub fn configure_process_group(cmd: &mut Command) {
    // SAFETY: setsid() is async-signal-safe. It creates a new session,
    // making this child the session leader with its own process group.
    unsafe {
        cmd.pre_exec(|| {
            if libc::setsid() == -1 {
                // Non-fatal: worst case we can only kill the top-level PID.
                // This should never fail in practice (child is not already
                // a session leader).
                eprintln!(
                    "warning: setsid() failed: {}",
                    std::io::Error::last_os_error()
                );
            }
            Ok(())
        });
    }
}

/// Spawn a command and wait for it with a timeout.
///
/// This is a convenience wrapper that configures the process group, sets up
/// piped stdout/stderr, spawns the child, and calls `wait_with_timeout`.
pub fn spawn_with_timeout(cmd: &mut Command, timeout_secs: u32) -> ProcessOutcome {
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    configure_process_group(cmd);

    match cmd.spawn() {
        Ok(child) => wait_with_timeout(child, timeout_secs),
        Err(e) => ProcessOutcome::SpawnError(format!("Failed to spawn: {}", e)),
    }
}

/// Wait for an already-spawned child process with a timeout.
///
/// Stdout and stderr are drained in background threads to prevent pipe
/// buffer deadlocks (the OS pipe buffer is ~64KB; a chatty child will
/// block on write() if nobody is reading).
///
/// Polls `try_wait()` every 100ms. On timeout:
/// 1. Sends `SIGKILL` via `killpg()` (kills entire process group).
/// 2. Waits 500ms for the process to die.
/// 3. If still alive, reads `/proc/<pid>/status` to confirm D-state.
///
/// The child MUST have been spawned with `configure_process_group()` for
/// `killpg()` to work correctly.
pub fn wait_with_timeout(mut child: Child, timeout_secs: u32) -> ProcessOutcome {
    let timeout = Duration::from_secs(timeout_secs as u64);
    let start = Instant::now();

    // Take pipe handles immediately and drain in background threads.
    // This prevents deadlock when a child produces >64KB of output.
    //
    // Uses read_to_end + from_utf8_lossy instead of read_to_string because
    // some tests (e.g. two_col) print raw uint8_t values that produce
    // invalid UTF-8 sequences. read_to_string would fail and return an
    // empty string, causing the runner to miss PASS/FAIL in the output.
    let stdout_thread = child.stdout.take().map(|pipe| {
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let mut reader = std::io::BufReader::new(pipe);
            let _ = reader.read_to_end(&mut bytes);
            String::from_utf8_lossy(&bytes).into_owned()
        })
    });
    let stderr_thread = child.stderr.take().map(|pipe| {
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let mut reader = std::io::BufReader::new(pipe);
            let _ = reader.read_to_end(&mut bytes);
            String::from_utf8_lossy(&bytes).into_owned()
        })
    });

    let join_threads = || {
        let stdout = stdout_thread
            .and_then(|t| t.join().ok())
            .unwrap_or_default();
        let stderr = stderr_thread
            .and_then(|t| t.join().ok())
            .unwrap_or_default();
        (stdout, stderr)
    };

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                // Process finished normally.
                let (stdout, stderr) = join_threads();
                return ProcessOutcome::Completed {
                    stdout,
                    stderr,
                    exit_code: exit_code_from_status(&status),
                };
            }
            Ok(None) => {
                // Still running -- check timeout.
                if start.elapsed() >= timeout {
                    return kill_and_assess(&mut child, join_threads);
                }
                std::thread::sleep(POLL_INTERVAL);
            }
            Err(e) => {
                return ProcessOutcome::SpawnError(
                    format!("Failed to wait for process: {}", e),
                );
            }
        }
    }
}

/// Kill the child's process group and determine if it survived (D-state).
///
/// `join_threads` collects output from the background drain threads.
fn kill_and_assess<F>(child: &mut Child, join_threads: F) -> ProcessOutcome
where
    F: FnOnce() -> (String, String),
{
    let pid = child.id();

    // Kill the entire process group (child is session leader from setsid).
    // Negative PID = process group kill. Falls back to single-PID kill
    // if killpg fails (e.g., setsid didn't work).
    let kill_result = unsafe { libc::killpg(pid as i32, libc::SIGKILL) };
    if kill_result == -1 {
        // killpg failed -- try direct kill as fallback.
        let _ = child.kill();
    }

    // Give the kernel time to deliver the signal and reap.
    std::thread::sleep(REAP_GRACE);

    // Killing the child closes its pipe ends, unblocking the drain threads.
    let (stdout, stderr) = join_threads();

    // Check if the process actually died.
    match child.try_wait() {
        Ok(Some(_)) => {
            // Dead. Normal timeout.
            ProcessOutcome::Timeout { stdout, stderr }
        }
        Ok(None) => {
            // Still alive after SIGKILL + 500ms. This is D-state.
            let state = check_proc_state(pid);
            log::error!(
                "Process {} survived SIGKILL (state: {}). Device is wedged.",
                pid, state
            );
            ProcessOutcome::Wedged { pid, stdout, stderr }
        }
        Err(_) => {
            // Error querying -- treat as normal timeout (conservative).
            ProcessOutcome::Timeout { stdout, stderr }
        }
    }
}

/// Read the process state character from `/proc/<pid>/status`.
///
/// Returns the state string (e.g. "D (disk sleep)") or "unknown" if
/// the proc entry is unreadable (process already gone, permissions, etc.).
fn check_proc_state(pid: u32) -> String {
    let status_path = format!("/proc/{}/status", pid);
    match std::fs::read_to_string(&status_path) {
        Ok(content) => {
            for line in content.lines() {
                if let Some(state) = line.strip_prefix("State:") {
                    return state.trim().to_string();
                }
            }
            "unknown (no State: line)".to_string()
        }
        Err(_) => "unknown (proc unreadable)".to_string(),
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completed_process() {
        let mut cmd = Command::new("echo");
        cmd.arg("hello");
        match spawn_with_timeout(&mut cmd, 5) {
            ProcessOutcome::Completed { stdout, exit_code, .. } => {
                assert_eq!(exit_code, 0);
                assert_eq!(stdout.trim(), "hello");
            }
            other => panic!("Expected Completed, got {:?}", outcome_name(&other)),
        }
    }

    #[test]
    fn test_nonzero_exit() {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "echo err >&2; exit 42"]);
        match spawn_with_timeout(&mut cmd, 5) {
            ProcessOutcome::Completed { stderr, exit_code, .. } => {
                assert_eq!(exit_code, 42);
                assert!(stderr.contains("err"));
            }
            other => panic!("Expected Completed, got {:?}", outcome_name(&other)),
        }
    }

    #[test]
    fn test_timeout_kills_process() {
        let mut cmd = Command::new("sleep");
        cmd.arg("60");
        let start = Instant::now();
        match spawn_with_timeout(&mut cmd, 1) {
            ProcessOutcome::Timeout { .. } => {
                // Should return in ~1s, not 60s.
                assert!(start.elapsed() < Duration::from_secs(5));
            }
            other => panic!("Expected Timeout, got {:?}", outcome_name(&other)),
        }
    }

    #[test]
    fn test_spawn_error() {
        let mut cmd = Command::new("/nonexistent/binary/that/does/not/exist");
        match spawn_with_timeout(&mut cmd, 5) {
            ProcessOutcome::SpawnError(_) => {}
            other => panic!("Expected SpawnError, got {:?}", outcome_name(&other)),
        }
    }

    #[test]
    fn test_process_group_kill() {
        // Spawn a shell that creates a subprocess, verify both are killed.
        // The shell starts a background sleep and waits -- if only the shell
        // is killed, the sleep would linger.
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "sleep 60 & wait"]);
        let start = Instant::now();
        match spawn_with_timeout(&mut cmd, 1) {
            ProcessOutcome::Timeout { .. } => {
                assert!(start.elapsed() < Duration::from_secs(5));
            }
            other => panic!("Expected Timeout, got {:?}", outcome_name(&other)),
        }
    }

    #[test]
    fn test_check_proc_state_self() {
        // Our own process should be in R (running) state.
        let pid = std::process::id();
        let state = check_proc_state(pid);
        // Should start with R or S (sleeping is also common for test runner).
        assert!(
            state.starts_with('R') || state.starts_with('S'),
            "unexpected state for self: {}",
            state
        );
    }

    #[test]
    fn test_check_proc_state_nonexistent() {
        // PID 0 is the idle process, but a very high PID should not exist.
        let state = check_proc_state(4_000_000_000);
        assert!(state.contains("unknown"));
    }

    /// Helper to name outcomes for panic messages in tests.
    fn outcome_name(outcome: &ProcessOutcome) -> &'static str {
        match outcome {
            ProcessOutcome::Completed { .. } => "Completed",
            ProcessOutcome::Timeout { .. } => "Timeout",
            ProcessOutcome::Wedged { .. } => "Wedged",
            ProcessOutcome::SpawnError(_) => "SpawnError",
        }
    }
}
