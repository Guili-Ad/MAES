"""Bounded diagnostics: no disk I/O in the perception/input hot path."""
from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .build_identity import VERSION, current_identity


class TapTrace:
    def __init__(self, config, capacity: int = 8192):
        self.run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid4().hex[:8]
        self.config = asdict(config)
        self.config_hash = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()[:16]
        self.identity = current_identity()
        self.calibration_hash = ''
        self.effective_calibration_hash = ''
        self.segment_id = 0
        self._critical = deque(maxlen=capacity)
        self._visual = deque(maxlen=min(capacity, 2048))
        self.dropped = 0
        self.critical_dropped = 0
        self.visual_dropped = 0
        self.visual_sampled_out = 0
        self._ordinal = 0
        self._visual_seen = {}
        self.summary = None

    def add(self, kind: str, **fields):
        # Full per-event changes remain critical. Repeated candidates/whole
        # frame marker inventories cannot evict an input or an identity change.
        visual = kind in {'flick_detected', 'hold_markers', 'head_component_merged',
                          'tap_contour_repeat'}
        if visual:
            key = (kind, fields.get('track'), fields.get('lane'))
            stamp = fields.get('time')
            previous = self._visual_seen.get(key)
            if isinstance(stamp, (float, int)) and previous is not None and 0 <= stamp-previous < .25:
                self.visual_sampled_out += 1
                return
            if len(self._visual_seen) >= 256 and key not in self._visual_seen:
                self._visual_seen.pop(next(iter(self._visual_seen)))
            if isinstance(stamp, (float, int)):
                self._visual_seen[key] = stamp
        buffer = self._visual if visual else self._critical
        if len(buffer) == buffer.maxlen:
            self.dropped += 1
            if visual:
                self.visual_dropped += 1
            else:
                self.critical_dropped += 1
        self._ordinal += 1
        buffer.append((self._ordinal, {'kind': kind, 'segment_id': self.segment_id, **fields}))

    @property
    def records(self):
        # Merge only during diagnostics/replay export, never in the hot loop.
        return [record for _, record in sorted((*self._critical, *self._visual), key=lambda item: item[0])]

    def write(self) -> Path:
        # Branch-local output; never writes the shared user calibration store.
        root = Path(__file__).resolve().parents[2] / 'logs' / 'tap-traces'
        root.mkdir(parents=True, exist_ok=True)
        path = root / (self.run_id + '.jsonl')
        header = {'schema': 4, **self.identity, 'run_id': self.run_id,
                  'calibration_hash': self.calibration_hash,
                  'effective_calibration_hash': self.effective_calibration_hash,
                  'config_hash': self.config_hash, 'config': self.config,
                  'dropped_records': self.dropped,
                  'critical_dropped': self.critical_dropped, 'visual_dropped': self.visual_dropped,
                  'visual_sampled_out': self.visual_sampled_out,
                  'clock': 'host perf_counter; not game judgement time'}
        if self.summary is not None:
            # Independent per-run evidence; music_last_result remains the
            # backwards-compatible latest-result view and may be overwritten.
            header['summary'] = self.summary
        with path.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(header, ensure_ascii=False) + '\n')
            for record in self.records:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        return path
