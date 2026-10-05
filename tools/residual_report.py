"""Fail closed on every fixture's admission guards; do not infer game BM."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

APP = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def p95(values):
    return sorted(values)[max(0, math.ceil(len(values)*.95)-1)] if values else 0.


def performance_pair(bases, candidates, files):
    """Pool every predeclared repetition; never discard a slow trial."""
    if not bases or len(bases) != len(candidates):
        raise ValueError('Complete matched repetitions required')
    guards = all(r['comparison']['comparison_valid']
                 and all(r['comparison']['guards'].values()) for r in candidates)
    for side in (bases, candidates):
        stable = ('input_digest', 'candidate_digest', 'config_hash', 'effective_calibration_hash',
                  'inventory_hash', 'fixture_file_hashes', 'source_hash', 'source_files',
                  'measurement_hash', 'tool_hash', 'frames')
        guards = guards and all(all(r.get(k) == side[0].get(k) for k in stable) for r in side)
    def delta(left, right):
        def values(reports, field):
            if field == 'joined':
                return [r['qualification_cpu_ms']+r['nonwaiting_planning_cpu_ms']+
                        r['nonwaiting_dispatch_cpu_ms'] for report in reports for r in report['rows']]
            return [r[field] for report in reports for r in report['rows']]
        return {name: p95(values(right, field))-p95(values(left, field)) for name, field in
                [('qualification', 'qualification_cpu_ms'), ('qualification_planning_dispatch', 'joined'),
                 ('frame_processing', 'frame_processing_cpu_ms')]}
    deltas = delta(bases, candidates)
    current = all(files == r['source_files'] for r in candidates)
    return dict(frames=candidates[0]['frames'], measured_samples=sum(r['frames'] for r in candidates),
        repetitions=len(candidates), guards=guards, source_current=current,
        source_hash=candidates[0]['source_hash'], guard_count=len(candidates[0]['comparison']['guards']),
        p95_delta_ms={k: round(v, 3) for k, v in deltas.items()},
        trial_p95_deltas_ms=[{k: round(v, 3) for k, v in delta([b], [c]).items()}
                            for b, c in zip(bases, candidates)],
        passed=guards and current and max(deltas.values()) <= 2.000001)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--additional-baselines', nargs='*', default=[])
    parser.add_argument('--additional-candidates', nargs='*', default=[])
    parser.add_argument('--contract', type=Path)
    parser.add_argument('--loop-baseline')
    parser.add_argument('--loop-candidate')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(args.additional_baselines) != len(args.additional_candidates):
        parser.error('Matched additional baseline/candidate labels required')
    baseline_labels = [args.baseline, *args.additional_baselines]
    candidate_labels = [args.candidate, *args.additional_candidates]
    if len(set(baseline_labels+candidate_labels)) != len(baseline_labels+candidate_labels):
        parser.error('Every repetition needs a distinct saved label')
    work = args.work_root.resolve()
    if not args.output.resolve().is_relative_to(work):
        parser.error('Evidence stays under the explicit work directory')
    if args.output.exists() or args.output.with_suffix('.md').exists():
        raise FileExistsError('Preserve prior reports')
    files = {str(p.relative_to(APP)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((APP/'agent').rglob('*.py'))}
    rows = read(work/'fixtures/clips.json')
    performance, failures, loops = [], [], []
    for fixture in rows:
        name = fixture['name']
        bases = [read(work/f'reports/{label}/{name}.json') for label in baseline_labels]
        candidates = [read(work/f'reports/{label}/{name}.json') for label in candidate_labels]
        item = performance_pair(bases, candidates, files)
        performance.append({'clip': name, **item})
        if not item['passed']:
            failures.append(f'Performance admission: {name}')
    if args.loop_baseline or args.loop_candidate:
        if not (args.loop_baseline and args.loop_candidate):
            parser.error('Both loop labels required')
        summaries = [read(work/f'loop-reports/{label}/summary.json') for label in
                     (args.loop_baseline, args.loop_candidate)]
        if not all(s['full_prelude'] and len(s['clips']) == len(rows) for s in summaries):
            failures.append('Incomplete full-prelude replay inventory')
        if summaries[1]['source_files'] != files:
            failures.append('Loop source no longer current')
        for fixture in rows:
            name = fixture['name']
            reports = [read(work/f'loop-reports/{label}/{name}.json') for label in
                       (args.loop_baseline, args.loop_candidate)]
            pair = []
            for report in reports:
                records = [r for r in report['trace'] if r['kind'] == 'input' and r.get('receipt')]
                identities = [(r.get('segment_id'), r['receipt']['event_id']) for r in records]
                target = [r for r in records if fixture['start'] <=
                          (r['receipt'].get('down_call_started') or -1) < fixture['end']]
                interesting = []
                for r in target:
                    if r.get('origin') not in {'hold_note', 'flick'}:
                        continue
                    down = r['receipt'].get('down_call_started')
                    latest = r.get('latest_visual_time')
                    interesting.append({'event': r['receipt']['event_id'], 'origin': r.get('origin'),
                        'visual_family': r.get('visual_family'), 'timing_profile': r.get('timing_profile'),
                        'physical_id': r.get('physical_id'),
                        'lane': r['receipt']['lane'], 'down': down, 'deadline': r.get('deadline'),
                        'late_ms': round((down-r['deadline'])*1000., 3) if down is not None else None,
                        'visual_age_ms': round((down-latest)*1000., 3) if down is not None and latest is not None else None})
                pair.append({'frames': report['frames'], 'captures': report['captures'],
                    'inputs': len(records), 'target_inputs': len(target), 'target_gold_flick_inputs': interesting,
                    'duplicate_receipts': len(identities)-len(set(identities)),
                    'input_errors': sum(bool(r['receipt'].get('error')) for r in records),
                    'stop': report['stop'], 'trace_buffer': report.get('trace_buffer'),
                    'metrics': report['metrics_ms'].get('whole_run', {})})
            if (reports[0]['config_hash'] != reports[1]['config_hash']
                    or reports[0]['effective_calibration_hash'] != reports[1]['effective_calibration_hash']
                    or reports[0]['cost_profile'] != reports[1]['cost_profile']):
                failures.append(f'Loop input mismatch: {name}')
            for item in pair:
                if item['duplicate_receipts'] or item['input_errors'] or item['stop']['cleanup_failure']:
                    failures.append(f'Loop receipt/cleanup failure: {name}')
                if item['trace_buffer'] is None or item['trace_buffer']['critical_dropped']:
                    failures.append(f'Loop critical trace incomplete: {name}')
            loops.append({'clip': name, 'baseline': pair[0], 'candidate': pair[1]})
    contract_path = args.contract or work/'final-contract-pixel.json'
    contract_report = read(contract_path)
    contract = contract_report['comparison']
    current_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    contract_current = (contract_report['candidate']['source_stable']
        and contract_report['candidate']['source_fingerprint'] == current_hash)
    if not contract['passed'] or not contract_current:
        failures.append('Protected action contracts failed')
    result = {'schema': 1, 'passed': not failures, 'failures': failures,
        'performance_labels': {'baseline': baseline_labels, 'candidate': candidate_labels},
        'pooling': 'All explicitly declared repetitions pooled; individual trial deltas retained; no slow trials discarded.',
        'protected_contract_current': contract_current,
        'performance': performance, 'loops': loops, 'protected_actions': contract,
        'limitations': 'Guarded fixed-PTS wall costs and production-loop median-cost mocks; no live controller or game judgment inference. Visual sampling/truncation is explicit; event ID uniqueness is not physical uniqueness.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    table = ['# 本地验证汇总', '', f'准入：{result["passed"]}；失败项：{failures}', '',
             '| 片段 | 帧数 | 守卫 | 资格P95增量ms | 资格＋规划＋派发 | 全帧增量 | 通过 |',
             '|---|---:|---:|---:|---:|---:|---|']
    for r in performance:
        d = r['p95_delta_ms']
        table.append(f'| {r["clip"]} | {r["frames"]} | {r["guard_count"]} | {d["qualification"]} | '
                     f'{d["qualification_planning_dispatch"]} | {d["frame_processing"]} | {r["passed"]} |')
    table += ['', f'每段重复次数：{performance[0]["repetitions"]}；所有声明的重复均进入分位统计，不剔除慢轮次。',
              '', result['limitations'], '', '完整逐窗口金圈／划动主机输入及观测年龄见同名JSON；不等于游戏Bad／Miss。']
    args.output.with_suffix('.md').write_text('\n'.join(table)+'\n', encoding='utf-8')
    print(json.dumps({'passed': result['passed'], 'failures': failures, 'frames': sum(r['frames'] for r in performance),
                      'measured_samples_per_source': sum(r['measured_samples'] for r in performance),
                      'loop_clips': len(loops)}, ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
