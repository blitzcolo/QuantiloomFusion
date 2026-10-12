"""Renderer process adapter: the only module that starts Quantiloom."""

import subprocess
import sys
from pathlib import Path
from subprocess import CalledProcessError  # re-exported for callers' error handling

from ..core.paths import canonical_path


class RenderProcess:
    """A running renderer process."""

    def __init__(self, popen):
        self._popen = popen
        self.pid = popen.pid

    def wait(self) -> int:
        return self._popen.wait()

    def terminate(self, timeout_s: float = 10.0) -> None:
        self._popen.terminate()
        try:
            self._popen.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self._popen.kill()
            self._popen.wait()


class QuantiloomCli:
    """The Windows Quantiloom CLI; on WSL it is reached through interop."""

    def __init__(self, cli: Path):
        self.cli = cli

    def version(self) -> str:
        return subprocess.check_output([str(self.cli), "--version"], text=True).strip()

    def start(self, job_toml: Path, workdir: Path, log) -> RenderProcess:
        return RenderProcess(
            subprocess.Popen(
                [str(self.cli), canonical_path(job_toml)],
                cwd=workdir,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        )


def run_closure_audit(script: Path, attempt: Path, timeout_s: float = 300) -> str:
    proc = subprocess.run(
        [sys.executable, str(script), str(attempt)],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return (proc.stdout + proc.stderr).strip()
