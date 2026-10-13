"""Metric interface and the append-only experiment result store."""

from __future__ import annotations

import json
import math
import os
from abc import ABC, abstractmethod
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

from ..core.locking import runner_lock
from ..dataset.types import Sample
from ..fusion.base import FusionInput, FusionOutput


class Metric(ABC):
    """Scores a fusion output against its input and the sample's ground truth."""

    name: str
    version: int

    @abstractmethod
    def __call__(self, out: FusionOutput, x: FusionInput, sample: Sample) -> dict[str, float]: ...


@dataclass(frozen=True)
class ResultKey:
    """Identity of one metric result row."""

    dataset_id: str
    sample_id: str
    method_fingerprint: str
    metric: str
    metric_version: int


class ResultStore:
    """Append-only JSONL result store: LF newlines, existing keys are skipped.

    One line per row: ``{"key": {...}, "values": {name: float | null}}``,
    UTF-8, compact, sorted keys.  Non-finite values are written as ``null``
    and read back as NaN.  A corrupt final line (crash mid-write) is dropped
    and the file truncated back to the last complete line; a corrupt earlier
    line raises ValueError with its line number.  The store takes an
    exclusive lock on its directory for its lifetime — a second ResultStore
    on the same directory raises ValueError (the lock helper's filenames and
    message text still say "generator", a cosmetic mismatch).
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._rows: dict[ResultKey, dict[str, float]] = {}
        self._closed = False
        self._stack = ExitStack()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stack.enter_context(runner_lock(self.path.parent))
        self._load()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._stack.close()

    def __enter__(self) -> "ResultStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _decode_line(line: bytes, lineno: int) -> tuple[ResultKey, dict[str, float]]:
        try:
            rec = json.loads(line.decode("utf-8"))
            k = rec["key"]
            key = ResultKey(
                str(k["dataset_id"]), str(k["sample_id"]),
                str(k["method_fingerprint"]), str(k["metric"]),
                int(k["metric_version"]),
            )
            values = {
                str(name): float("nan") if v is None else float(v)
                for name, v in rec["values"].items()
            }
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            raise ValueError(f"{lineno}: corrupt result row: {e}") from e
        return key, values

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        lines = raw.split(b"\n")
        if lines and lines[-1] == b"":
            lines.pop()  # final LF is a terminator, not an empty row
        offset = 0
        corrupt_tail = None
        for i, line in enumerate(lines, start=1):
            start = offset
            offset += len(line) + 1
            if not line:
                raise ValueError(f"{self.path}:{i}: empty row")
            try:
                key, values = self._decode_line(line, i)
            except ValueError as e:
                if i == len(lines):
                    corrupt_tail = (i, start, e)
                    break
                raise ValueError(f"{self.path}:{e}") from e
            self._rows[key] = values
        if corrupt_tail is not None:
            _, start, _ = corrupt_tail
            # Repair now, while the lock is held, so the next append cannot
            # land after a torn row.
            with self.path.open("r+b") as f:
                f.truncate(start)

    def has(self, key: ResultKey) -> bool:
        return key in self._rows

    def get(self, key: ResultKey) -> dict[str, float] | None:
        values = self._rows.get(key)
        return None if values is None else dict(values)

    def rows(self):
        """Yield ``(ResultKey, values)`` in file order."""
        yield from self._rows.items()

    def append(self, key: ResultKey, values: dict[str, float]) -> None:
        if self._closed:
            raise ValueError("result store is closed")
        if key in self._rows:
            raise ValueError(f"duplicate result key {key}")
        clean = {
            str(name): float(v) if math.isfinite(float(v)) else None
            for name, v in values.items()
        }
        line = json.dumps(
            {"key": asdict(key), "values": clean},
            ensure_ascii=True, allow_nan=False,
            sort_keys=True, separators=(",", ":"),
        ) + "\n"
        with self.path.open("ab") as f:
            f.write(line.encode("ascii"))
            f.flush()
            os.fsync(f.fileno())
        self._rows[key] = {
            name: (float("nan") if v is None else v) for name, v in clean.items()
        }
