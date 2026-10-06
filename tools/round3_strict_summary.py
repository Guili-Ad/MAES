"""Guarded summary of identical-source-PTS wall-time qualification evidence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

APP_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = APP_ROOT.parent/'.work/round3-implementation-20261003/reports'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior evidence')
    root = APP_ROOT/'temp/round3-implementation'
    summary = json.loads((root/args.candidate/'summary.json').read_text())
    rows = []
    for entry in summary:
        name = entry['clip']['name']
        old = json.loads((root/args.baseline/f'{name}.json').read_text())
        new = json.loads((root/args.candidate/f'{name}.json').read_text())
        cmp = new['comparison']
        rows.append({'name':name,'frames':new['frames'],
            'baseline_qualification_wall_p95_ms':old['timing_ms']['p95'],
            'candidate_qualification_wall_p95_ms':new['timing_ms']['p95'],
            'qualification_wall_p95_increment_ms':cmp['p95_increment_ms'],
            'guards':cmp['guards'],'comparison_valid':cmp['comparison_valid'],
            'guarded_2ms_pass':cmp['within_2ms'],
            'planning_wall_p95_increment_ms':cmp['planning_p95_increment_ms'],
            'dispatch_wall_p95_increment_ms':cmp['dispatch_p95_increment_ms'],
            'frame_processing_wall_p95_increment_ms':cmp['frame_processing_p95_increment_ms'],
            'baseline_auxiliary_process_time':old['process_cpu_timing_ms'],
            'candidate_auxiliary_process_time':new['process_cpu_timing_ms'],
            'source_hash':new['source_hash'],'tool_hash':new['tool_hash'],
            'measurement_hash':new['measurement_hash'],'trace_dropped':new['trace_dropped'],
            'predecode':new['predecode']})
    report = {'schema':2,'mode':'guarded-identical-source-PTS-wall-qualification-summary',
        'baseline_label':args.baseline,'candidate_label':args.candidate,'clips':rows,
        'clips_count':len(rows),'frames':sum(x['frames'] for x in rows),
        'all_guards_pass':all(x['comparison_valid'] for x in rows),
        'all_qualification_wall_pass':all(x['guarded_2ms_pass'] for x in rows),
        'max_qualification_wall_p95_increment_ms':max(x['qualification_wall_p95_increment_ms'] for x in rows),
        'max_planning_wall_p95_increment_ms':max(x['planning_wall_p95_increment_ms'] for x in rows),
        'max_dispatch_wall_p95_increment_ms':max(x['dispatch_wall_p95_increment_ms'] for x in rows),
        'frame_processing_wall_decreased_all_clips':all(x['frame_processing_wall_p95_increment_ms']<0 for x in rows),
        'source_hashes':sorted({x['source_hash'] for x in rows}),
        'tool_hashes':sorted({x['tool_hash'] for x in rows}),
        'measurement_hashes':sorted({x['measurement_hash'] for x in rows}),
        'notes':'perf_counter elapsed wall gate retained. process_time only auxiliary; Windows granularity can quantize short samples, not a replacement gate. One clip is predecoded in memory and FFmpeg exits before measurement. Every decoded frame/PTS consumed once with instant mock transport and real head receipts; no song actions injected. Qualification excludes provider/mask/OCR/input/release/refine; planning/dispatch/frame-processing are separately reported to avoid hiding shifted work. Not a full production-loop/input-latency/BM/FC approval. Physical identity/action-count alignment is not inferred by ordinal.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    shutil.copy2(args.output,REPORT_ROOT/args.output.name)
    print(json.dumps({key:report[key] for key in ('clips_count','frames','all_guards_pass','all_qualification_wall_pass','max_qualification_wall_p95_increment_ms','max_planning_wall_p95_increment_ms','max_dispatch_wall_p95_increment_ms','frame_processing_wall_decreased_all_clips')}))
    return 0 if report['all_qualification_wall_pass'] else 1


if __name__=='__main__':
    raise SystemExit(main())
