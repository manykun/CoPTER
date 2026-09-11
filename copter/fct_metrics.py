"""Incremental flow-completion-time measurements for an ns-3 episode."""

import csv
import math
from collections import deque
from pathlib import Path


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class FCTStepTracker:
    """Tail an ns-3 FCT file and emit one JSON record per environment step.

    A flow's FCT becomes observable only when the flow completes.  Therefore
    ``step_*`` fields describe flows completed since the preceding environment
    step; ``rolling_*`` fields use the latest completed flows and are intended
    for readable curves.  This tracker never changes the environment, policy,
    replay, reward, or optimizer.
    """

    def __init__(
        self,
        source_path="",
        trace_path="",
        *,
        episode=None,
        phase="",
        eval_tag="",
        rolling_flows=256,
    ):
        self.source_path = Path(source_path).expanduser() if source_path else None
        self.trace_path = Path(trace_path).expanduser() if trace_path else None
        self.episode = episode
        self.phase = phase
        self.eval_tag = eval_tag
        self.rolling_flows = int(rolling_flows)
        if self.rolling_flows <= 0:
            raise ValueError("rolling_flows must be positive")
        if bool(self.source_path) != bool(self.trace_path):
            raise ValueError("FCT source and step-trace paths must be set together")
        self._offset = 0
        self._remainder = ""
        self._rolling = deque(maxlen=self.rolling_flows)
        self._cumulative_count = 0
        self._cumulative_bytes = 0
        self._cumulative_fct_us = 0.0
        self._cumulative_slowdown = 0.0
        self._invalid_lines = 0
        self._records = 0
        self._handle = None
        self._writer = None
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            # The launcher gives every episode its own path.  Re-running an
            # interrupted episode replaces, rather than duplicates, its rows.
            self._handle = self.trace_path.open(
                "w", encoding="utf-8", newline="", buffering=1
            )

    @property
    def enabled(self):
        return self._handle is not None

    def _read_new_flows(self):
        if not self.enabled or not self.source_path.exists():
            return []
        with self.source_path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
            self._offset = handle.tell()
        text = self._remainder + chunk
        lines = text.splitlines(keepends=True)
        self._remainder = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._remainder = lines.pop()
        flows = []
        for line in lines:
            fields = line.split()
            if len(fields) < 8:
                self._invalid_lines += 1
                continue
            try:
                size = int(fields[4])
                fct_ns = float(fields[6])
                ideal_ns = float(fields[7])
            except ValueError:
                self._invalid_lines += 1
                continue
            if fct_ns <= 0 or ideal_ns <= 0:
                self._invalid_lines += 1
                continue
            flows.append((size, fct_ns / 1000.0, fct_ns / ideal_ns))
        return flows

    def observe(self, local_step, *, global_env_step=None, global_train_step=None, terminal=False):
        if not self.enabled:
            return
        flows = self._read_new_flows()
        fcts = [item[1] for item in flows]
        slowdowns = [item[2] for item in flows]
        for item in flows:
            self._rolling.append(item)
        count = len(flows)
        step_bytes = sum(item[0] for item in flows)
        step_fct_sum = sum(fcts)
        step_slowdown_sum = sum(slowdowns)
        self._cumulative_count += count
        self._cumulative_bytes += step_bytes
        self._cumulative_fct_us += step_fct_sum
        self._cumulative_slowdown += step_slowdown_sum
        rolling_fcts = [item[1] for item in self._rolling]
        rolling_slowdowns = [item[2] for item in self._rolling]
        record = {
            "schema_version": 1,
            "episode": self.episode,
            "phase": self.phase,
            "eval_tag": self.eval_tag,
            "local_step": int(local_step),
            "global_env_step": None if global_env_step is None else int(global_env_step),
            "global_train_step": None if global_train_step is None else int(global_train_step),
            "terminal": bool(terminal),
            "step_completed_flows": count,
            "step_bytes": step_bytes,
            "step_fct_sum_us": step_fct_sum,
            "step_mean_fct_us": step_fct_sum / count if count else None,
            "step_p50_fct_us": _percentile(fcts, 0.50),
            "step_p95_fct_us": _percentile(fcts, 0.95),
            "step_p99_fct_us": _percentile(fcts, 0.99),
            "step_mean_slowdown": step_slowdown_sum / count if count else None,
            "step_p99_slowdown": _percentile(slowdowns, 0.99),
            "rolling_completed_flows": len(self._rolling),
            "rolling_mean_fct_us": (
                sum(rolling_fcts) / len(rolling_fcts) if rolling_fcts else None
            ),
            "rolling_p95_fct_us": _percentile(rolling_fcts, 0.95),
            "rolling_p99_slowdown": _percentile(rolling_slowdowns, 0.99),
            "cumulative_completed_flows": self._cumulative_count,
            "cumulative_bytes": self._cumulative_bytes,
            "cumulative_mean_fct_us": (
                self._cumulative_fct_us / self._cumulative_count
                if self._cumulative_count else None
            ),
            "cumulative_mean_slowdown": (
                self._cumulative_slowdown / self._cumulative_count
                if self._cumulative_count else None
            ),
            "invalid_fct_lines": self._invalid_lines,
        }
        if self._writer is None:
            self._writer = csv.DictWriter(self._handle, fieldnames=list(record))
            self._writer.writeheader()
        self._writer.writerow(record)
        self._records += 1

    def summary(self):
        return {
            "enabled": self.enabled,
            "source_file": str(self.source_path) if self.source_path else None,
            "trace_file": str(self.trace_path) if self.trace_path else None,
            "records": self._records,
            "completed_flows": self._cumulative_count,
            "mean_fct_us": (
                self._cumulative_fct_us / self._cumulative_count
                if self._cumulative_count else None
            ),
            "mean_slowdown": (
                self._cumulative_slowdown / self._cumulative_count
                if self._cumulative_count else None
            ),
            "invalid_lines": self._invalid_lines,
            "rolling_window_flows": self.rolling_flows,
        }

    def close(self):
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None
            self._writer = None
