"""RUN stop propagation (RUN-STOP-ARCH.md): the executor's poll loop
kills the process tree on stop_check and raises KeyboardInterrupt;
timeout still works; no stop_check → today's blocking behavior.

Real subprocesses, real timing — no mocks. Each test sleeps 30s in a
child; stop fires after ~1s, so each test runs ~1.5s."""

import subprocess
import sys
import time

import pytest

from agentic_core.actions import ActionExecutor


def _executor(stop_check=None, timeout=30):
    return ActionExecutor(workdir=".", run_timeout=timeout,
                          stop_check=stop_check)


class TestRunStop:
    def test_stop_kills_and_interrupts(self):
        """stop_check fires → tree killed, KeyboardInterrupt raised."""
        import threading
        stop = threading.Event()

        def check():
            return stop.is_set()

        ex = _executor(stop_check=check)
        action = {"type": "run", "code": "python -c \"import time; time.sleep(30)\"",
                  "lang": "bash"}
        t = threading.Timer(1.0, stop.set)
        t.start()
        start = time.monotonic()
        with pytest.raises(KeyboardInterrupt):
            ex._do_run(action)
        elapsed = time.monotonic() - start
        t.cancel()
        assert elapsed < 5, f"stop took {elapsed:.1f}s — poll loop not working"

    def test_timeout_still_works(self):
        """No stop; run_timeout expires → timeout observation, tree killed."""
        ex = _executor(stop_check=lambda: False, timeout=2)
        action = {"type": "run",
                  "code": "python -c \"import time; time.sleep(30)\"",
                  "lang": "bash"}
        start = time.monotonic()
        result = ex._do_run(action)
        elapsed = time.monotonic() - start
        assert "timed out after 2s" in result
        assert elapsed < 10

    def test_normal_completion(self):
        """Fast command completes normally through the poll loop."""
        ex = _executor(stop_check=lambda: False, timeout=30)
        action = {"type": "run", "code": "python -c \"print('hello')\"",
                  "lang": "bash"}
        result = ex._do_run(action)
        assert "hello" in result
        assert "[exit code: 0]" in result

    def test_tree_kill_kills_children(self):
        """A script that spawns a child — stop kills both (D3)."""
        import threading
        stop = threading.Event()
        ex = _executor(stop_check=lambda: stop.is_set(), timeout=30)
        # Parent sleeps 30s; child (spawned immediately) also sleeps 30s
        code = ("python -c \"import subprocess,time;"
                "subprocess.Popen(['python','-c','import time;time.sleep(30)']);"
                "time.sleep(30)\"")
        action = {"type": "run", "code": code, "lang": "bash"}
        t = threading.Timer(1.0, stop.set)
        t.start()
        with pytest.raises(KeyboardInterrupt):
            ex._do_run(action)
        t.cancel()
        time.sleep(0.5)  # let taskkill settle
        # No python sleeping-30s processes should remain
        check = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object {$_.CommandLine -like '*time.sleep(30)*'}).Count"],
            capture_output=True, text=True, timeout=15)
        count = int(check.stdout.strip() or "0")
        assert count == 0, f"{count} orphaned sleep processes remain"

    def test_stop_with_orphaned_pipe_holder(self):
        """`start`-style hang: a grandchild inherits the pipe write
        handles and outlives the tree kill. The drain must be BOUNDED —
        an unbounded communicate() would wait for the orphan's EOF
        forever and Stop would do nothing (the reported bug).
        start /b: no window, child inherits the pipes."""
        import threading
        stop = threading.Event()
        ex = _executor(stop_check=lambda: stop.is_set(), timeout=30)
        action = {"type": "run",
                  "code": "start /b python -c \"import time; time.sleep(30)\"",
                  "lang": "bash"}
        t = threading.Timer(1.0, stop.set)
        t.start()
        start = time.monotonic()
        with pytest.raises(KeyboardInterrupt):
            ex._do_run(action)
        elapsed = time.monotonic() - start
        t.cancel()
        assert elapsed < 8, (f"stop took {elapsed:.1f}s — the drain hung "
                             f"on orphan-held pipes")

    def test_event_object_not_truthy_stop(self):
        """Regression: Mercury wires stop_check from a closure that
        returns the Event OBJECT (StoppableClient's contract). An Event
        is always truthy — if the executor treats the return value as a
        bool directly, every RUN self-interrupts on the first poll.
        The wiring must wrap it: bool = event is not None and
        event.is_set(). This test reproduces the exact shape."""
        import threading
        events = {"k": threading.Event()}  # unset — no stop requested

        def get_event():          # StoppableClient's contract: the object
            return events["k"]

        # The CORRECT wrapper (what runtime.py must do)
        def stop_requested() -> bool:
            ev = get_event()
            return ev is not None and ev.is_set()

        ex = _executor(stop_check=stop_requested, timeout=10)
        action = {"type": "run", "code": "python -c \"print('ok')\"",
                  "lang": "bash"}
        result = ex._do_run(action)   # must complete, not interrupt
        assert "[exit code: 0]" in result

        # And the wrong wiring (the bug): passing the object-returning
        # closure directly — assert the contract violation is detectable
        assert bool(get_event()) is True  # an Event object IS truthy

    def test_no_stop_check_blocks_until_timeout(self):
        """stop_check=None → today's behavior: blocking until timeout."""
        ex = _executor(stop_check=None, timeout=2)
        action = {"type": "run",
                  "code": "python -c \"import time; time.sleep(30)\"",
                  "lang": "bash"}
        result = ex._do_run(action)
        assert "timed out after 2s" in result
