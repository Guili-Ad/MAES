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
        self.segment_id = 0
        self.records = deque(maxlen=capacity)
        self.dropped = 0
        self.summary = None

    def add(self, kind: str, **fields):
        if len(self.records) == self.records.maxlen:
            self.dropped += 1
        self.records.append({'kind': kind, 'segment_id': self.segment_id, **fields})

    def write(self) -> Path:
        # Branch-local output; never writes the shared user calibration store.
        root = Path(__file__).resolve().parents[2] / 'logs' / 'tap-traces'
        root.mkdir(parents=True, exist_ok=True)
        path = root / (self.run_id + '.jsonl')
        header = {'schema': 3, **self.identity, 'run_id': self.run_id,
                  'calibration_hash': self.calibration_hash,
                  'config_hash': self.config_hash, 'config': self.config,
                  'dropped_records': self.dropped, 'clock': 'host perf_counter; not game judgement time'}
        if self.summary is not None:
            # Independent per-run evidence; music_last_result remains the
            # backwards-compatible latest-result view and may be overwritten.
            header['summary'] = self.summary
        with path.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(header, ensure_ascii=False) + '\n')
            for record in self.records:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        return path
