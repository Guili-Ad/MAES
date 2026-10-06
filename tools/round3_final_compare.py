"""Compare the behavior-preserving performance stage, not different models."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

APP_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = APP_ROOT.parent/'.work/round3-implementation-20261003/reports'


def canonical(value):
    if isinstance(value, dict):
        return {key:canonical(item) for key,item in value.items() if key!='run_id'}
    if isinstance(value, list):
        return [canonical(item) for item in value]
    return value


def differences(old, new, path='', result=None):
    result = [] if result is None else result
    if type(old)!=type(new):
        result.append({'path':path,'old':old,'new':new})
    elif isinstance(old, dict):
        if set(old)!=set(new):
            result.append({'path':path,'old_keys':list(old),'new_keys':list(new)})
        for key in old.keys() & new.keys():
            differences(old[key],new[key],path+'/'+key,result)
    elif isinstance(old, list):
        if len(old)!=len(new):
            result.append({'path':path,'old_count':len(old),'new_count':len(new)})
        for index,(a,b) in enumerate(zip(old,new)):
            differences(a,b,path+'/'+str(index),result)
    elif old!=new:
        result.append({'path':path,'old':old,'new':new})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',default='final-loop')
    parser.add_argument('--after',default='final-loop-fast')
    parser.add_argument('--clips',nargs='*')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior comparison evidence')
    root = APP_ROOT/'temp/round3-implementation'
    summaries = {label:json.loads((root/label/'summary.json').read_text())
                 for label in (args.before,args.after)}
    names = [row['name'] for row in summaries[args.before]]
    if args.clips:
        names = [name for name in names if name in args.clips]
    records = []
    for name in names:
        old = json.loads((root/args.before/f'{name}.json').read_text())
        new = json.loads((root/args.after/f'{name}.json').read_text())
        fields = ('heads','actions','scheduled','pending_at_clip_end','observations','stop')
        field_equal = {key:old[key]==new[key] for key in fields}
        action_deltas = [abs(left['time']-right['time'])*1000 for left,right in zip(old['actions'],new['actions'])]
        action_structure_equal = (len(old['actions'])==len(new['actions']) and
            all({k:v for k,v in left.items() if k!='time'}=={k:v for k,v in right.items() if k!='time'}
                for left,right in zip(old['actions'],new['actions'])))
        trace_dif = differences(canonical(old['trace']),canonical(new['trace']))
        receipts = [x for x in new['trace'] if x['kind']=='input']
        old_inputs = {(row.get('segment_id'),row['event']):row for row in old['trace'] if row['kind']=='input'}
        new_inputs = {(row.get('segment_id'),row['event']):row for row in new['trace'] if row['kind']=='input'}
        matched_changes = []
        input_deadline_deltas = []
        input_call_deltas = []
        input_call_structure_equal = old_inputs.keys()==new_inputs.keys()
        for key in old_inputs.keys() & new_inputs.keys():
            left,right = canonical(old_inputs[key]),canonical(new_inputs[key])
            changes = differences(left,right)
            if changes:
                matched_changes.append({'segment':key[0],'event':key[1],'changes':changes})
            for field in ('deadline','raw_hit'):
                if left.get(field) is not None and right.get(field) is not None:
                    input_deadline_deltas.append(abs(left[field]-right[field])*1000)
            for field in ('down_call_started','down_call_finished','up_call_started','up_call_finished',
                          'move_call_started','move_call_finished'):
                a,b = left['receipt'].get(field),right['receipt'].get(field)
                if isinstance(a,list) and isinstance(b,list):
                    if len(a)!=len(b):
                        input_call_structure_equal = False
                    input_call_deltas.extend(abs(x-y)*1000 for x,y in zip(a,b))
                elif a is not None and b is not None:
                    input_call_deltas.append(abs(a-b)*1000)
                elif a!=b:
                    input_call_structure_equal = False
        timing_dif = [x for x in trace_dif if any(key in x['path'] for key in
                      ('deadline','raw_hit','call_started','call_finished','latest_visual_time'))]
        numeric_deltas = [abs(x['old']-x['new'])*1000 for x in timing_dif
                          if isinstance(x.get('old'),(int,float)) and isinstance(x.get('new'),(int,float))]
        errors = [x for x in receipts if x['receipt'].get('error')]
        up_missing = [x['event'] for x in receipts if x['receipt'].get('up_call_finished') is None]
        ids = Counter((x.get('segment_id'),x['event']) for x in receipts)
        record = {'name':name,'fields_exact':field_equal,
            'trace_exact_except_run_id':not trace_dif,'trace_differences':trace_dif[:15],
            'trace_difference_count':len(trace_dif),'max_input_or_deadline_difference_ms':max(numeric_deltas,default=0),
            'receipt_count':len(receipts),'origin_counts':dict(Counter(x['origin'] for x in receipts)),
            'action_kind_count_order_contact_coordinates_equal':action_structure_equal,
            'max_action_call_time_difference_ms':max(action_deltas,default=0),
            'actions_equivalent_01ms':action_structure_equal and max(action_deltas,default=0)<=.1,
            'matched_input_call_times_equivalent_01ms':input_call_structure_equal and max(input_call_deltas,default=0)<=.1,
            'max_matched_input_call_time_difference_ms':max(input_call_deltas,default=0),
            'input_event_keys_equal':old_inputs.keys()==new_inputs.keys(),
            'missing_input_event_ids':[list(key) for key in old_inputs.keys()-new_inputs.keys()],
            'added_input_event_ids':[list(key) for key in new_inputs.keys()-old_inputs.keys()],
            'matched_input_changes':matched_changes,
            'max_matched_input_deadline_or_raw_hit_difference_ms':max(input_deadline_deltas,default=0),
            'receipt_errors':errors,'up_missing':up_missing,
            'duplicate_event_ids':[list(key) for key,count in ids.items() if count>1],
            'pass':all(field_equal.values()) and not trace_dif and not errors and not up_missing}
        records.append(record)
    report = {'schema':1,'before':args.before,'after':args.after,'clips':records,
        'source_before':sorted({x['source_before']['sha256'] for x in summaries[args.before]}),
        'source_after':sorted({x['source_before']['sha256'] for x in summaries[args.after]}),
        'all_pass':all(x['pass'] for x in records),
        'all_actions_equivalent_01ms':all(x['actions_equivalent_01ms'] and x['matched_input_call_times_equivalent_01ms'] for x in records),
        'max_input_or_deadline_difference_ms':max(x['max_input_or_deadline_difference_ms'] for x in records),
        'receipts':sum(x['receipt_count'] for x in records),
        'notes':'Identical fixed cost profile; exact action equality confirms same production-loop source phase. If actions/identities differ, ordinal trace differences and their max delta are diagnostic only; use matched stable input event keys for meaningful changes and identify any approved correctness fix separately. No claim that differing physical IDs correspond to identical musical notes. Only random run_id is excluded. CPU fields deliberately excluded. Host mock input times are not game judgement times. No offline BM/FC inference.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    shutil.copy2(args.output,REPORT_ROOT/args.output.name)
    print(json.dumps({key:report[key] for key in ('all_pass','receipts','max_input_or_deadline_difference_ms','source_after')}))
    return 0 if report['all_pass'] else 1


if __name__=='__main__':
    raise SystemExit(main())
