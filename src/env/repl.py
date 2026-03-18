"""Persistent REPL for sandboxed code execution across episode steps.

Supports Docker containers (primary) and local subprocess fallback.
State persists via a cumulative script that grows each step.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod

from loguru import logger

from src.env.tools import TOOL_PREAMBLE

MAX_OUTPUT_CHARS = 8000


class BaseREPL(ABC):
    """Abstract interface for a persistent REPL session."""

    @abstractmethod
    def start_session(self) -> None: ...

    @abstractmethod
    def execute(self, code: str, timeout: int = 30) -> str: ...

    @abstractmethod
    def kill_session(self) -> None: ...


class DockerREPL(BaseREPL):
    """Persistent REPL using a Docker container with cumulative script."""

    def __init__(
        self,
        image: str = "rlm-sandbox",
        memory_limit: str = "512m",
        corpus_path: str = "data/corpus",
    ) -> None:
        self.image = image
        self.memory_limit = memory_limit
        self.corpus_path = os.path.abspath(corpus_path)
        self._container_id: str | None = None
        self._cumulative_script: str = ""
        self._step: int = 0

    def start_session(self) -> None:
        """Create and start a Docker container, inject tool preamble."""
        corpus_mount = f"{self.corpus_path}:/workspace/corpus:ro"
        result = subprocess.run(
            [
                "docker", "create",
                "--memory", self.memory_limit,
                "--network", "none",
                "-v", corpus_mount,
                "-i", self.image,
                "python3", "-i",
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Docker create failed: {result.stderr}")

        self._container_id = result.stdout.strip()
        subprocess.run(
            ["docker", "start", self._container_id],
            capture_output=True, text=True, timeout=10,
        )
        logger.info(f"Docker container started: {self._container_id[:12]}")

        self._cumulative_script = TOOL_PREAMBLE
        self._step = 0
        output = self._run_script(self._cumulative_script)
        logger.debug(f"Preamble output: {output[:200]}")

    def execute(self, code: str, timeout: int = 30) -> str:
        """Append code to cumulative script and execute the full script."""
        if self._container_id is None:
            raise RuntimeError("No active session — call start_session() first")

        self._step += 1
        self._cumulative_script += f"\n# --- Step {self._step} ---\n{code}\n"
        return self._run_script(self._cumulative_script, timeout=timeout)

    def _run_script(self, script: str, timeout: int = 30) -> str:
        """Write script to container and execute it."""
        assert self._container_id is not None

        # Write script to temp file inside container
        escaped = script.replace("'", "'\\''")
        subprocess.run(
            ["docker", "exec", "-i", self._container_id,
             "sh", "-c", f"cat > /tmp/step.py << 'SCRIPT_EOF'\n{script}\nSCRIPT_EOF"],
            capture_output=True, text=True, timeout=10,
        )

        # Execute the script
        try:
            result = subprocess.run(
                ["docker", "exec", "-i", self._container_id,
                 "python3", "/tmp/step.py"],
                capture_output=True, text=True, timeout=timeout,
            )
            output = result.stdout
            if result.stderr:
                output += "\nSTDERR:\n" + result.stderr
        except subprocess.TimeoutExpired:
            output = f"ERROR: Execution timed out after {timeout}s"

        return output[:MAX_OUTPUT_CHARS]

    def kill_session(self) -> None:
        """Remove the Docker container."""
        if self._container_id:
            subprocess.run(
                ["docker", "rm", "-f", self._container_id],
                capture_output=True, text=True, timeout=10,
            )
            logger.info(f"Docker container removed: {self._container_id[:12]}")
            self._container_id = None
            self._cumulative_script = ""
            self._step = 0


class LocalREPL(BaseREPL):
    """Persistent REPL using a local subprocess with cumulative script.

    Fallback when Docker is unavailable.
    """

    def __init__(self, corpus_path: str = "data/corpus") -> None:
        self.corpus_path = os.path.abspath(corpus_path)
        self._cumulative_script: str = ""
        self._step: int = 0
        self._tmpdir: tempfile.TemporaryDirectory | None = None

    def start_session(self) -> None:
        """Initialize the local REPL session with tool preamble."""
        self._tmpdir = tempfile.TemporaryDirectory()
        self._cumulative_script = TOOL_PREAMBLE
        self._step = 0

        output = self._run_script(self._cumulative_script)
        logger.info("Local REPL session started")
        logger.debug(f"Preamble output: {output[:200]}")

    def execute(self, code: str, timeout: int = 30) -> str:
        """Append code to cumulative script and execute."""
        self._step += 1
        self._cumulative_script += f"\n# --- Step {self._step} ---\n{code}\n"
        return self._run_script(self._cumulative_script, timeout=timeout)

    def _run_script(self, script: str, timeout: int = 30) -> str:
        """Write cumulative script to temp file and execute."""
        if self._tmpdir is None:
            raise RuntimeError("No active session — call start_session() first")

        script_path = os.path.join(self._tmpdir.name, "step.py")
        with open(script_path, "w") as f:
            f.write(script)

        env = os.environ.copy()
        env["CORPUS_DIR"] = self.corpus_path

        try:
            result = subprocess.run(
                ["python3", script_path],
                capture_output=True, text=True, timeout=timeout,
                env=env,
            )
            output = result.stdout
            if result.stderr:
                output += "\nSTDERR:\n" + result.stderr
        except subprocess.TimeoutExpired:
            output = f"ERROR: Execution timed out after {timeout}s"

        return output[:MAX_OUTPUT_CHARS]

    def kill_session(self) -> None:
        """Clean up temp directory."""
        if self._tmpdir:
            self._tmpdir.cleanup()
            self._tmpdir = None
            self._cumulative_script = ""
            self._step = 0
            logger.info("Local REPL session ended")


class PersistentREPL:
    """Auto-selecting REPL — uses Docker if available, falls back to local."""

    def __init__(
        self,
        use_docker: bool | None = None,
        image: str = "rlm-sandbox",
        memory_limit: str = "512m",
        corpus_path: str = "data/corpus",
    ) -> None:
        if use_docker is None:
            use_docker = self._docker_available(image)

        if use_docker:
            logger.info("Using Docker REPL")
            self._impl: BaseREPL = DockerREPL(
                image=image, memory_limit=memory_limit, corpus_path=corpus_path,
            )
        else:
            logger.info("Docker unavailable — using local REPL fallback")
            self._impl = LocalREPL(corpus_path=corpus_path)

    @staticmethod
    def _docker_available(image: str) -> bool:
        """Check if Docker is running and the sandbox image exists."""
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", image],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def start_session(self) -> None:
        self._impl.start_session()

    def execute(self, code: str, timeout: int = 30) -> str:
        return self._impl.execute(code, timeout=timeout)

    def kill_session(self) -> None:
        self._impl.kill_session()
