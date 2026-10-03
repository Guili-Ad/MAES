"""Summarize development replay cProfile without conflating CPU and latency."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import pstats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a distinct output name to preserve evidence')
    text = io.StringIO()
    stats = pstats.Stats(str(args.profile),stream=text)
    stats.sort_stats('cumulative').print_stats(30)
    records = []
    for (file,line,function),(primitive,calls,own,cumulative,callers) in stats.stats.items():
        records.append({'file':file,'line':line,'function':function,'primitive_calls':primitive,
            'calls':calls,'own_profiled_seconds':own,'cumulative_profiled_seconds':cumulative,
            'inclusive_fraction_of_profiled_elapsed':cumulative/stats.total_tt if stats.total_tt else None})
    records.sort(key=lambda row:row['cumulative_profiled_seconds'],reverse=True)
    selected = [row for row in records if row['function'] in
                ('refresh','detect_hold_tails','build_gold_mask','bonus_hold_ribbon_present',
                 'bonus_hold_ribbon_presence_batch','_bind_flicks','bind_flicks','recover_missing','update')
                and '/agent/music/' in row['file'].replace('\\','/')]
    report = {'schema':2,'profile':str(args.profile),'total_profiled_seconds':stats.total_tt,
        'top30_cumulative':records[:30],'selected_hotspots':selected,
        'notes':'cProfile default elapsed timer, not process_time CPU: read/join/controller waits and scheduling can be included. Instrumentation changes absolute timing; hotspot evidence only, not P95 gate. Inclusive cumulative function fractions overlap and must not be added. Strict fixture mock transport instant; profile includes video decode/provider and fixture/report overhead.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    args.output.with_suffix('.txt').write_text(text.getvalue(),encoding='utf-8')
    print(json.dumps({'output':str(args.output),'total_profiled_seconds':stats.total_tt,
                      'selected':selected},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
