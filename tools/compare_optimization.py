"""Compare protected action contracts with explicit 0.1 ms timing tolerance."""
import argparse
import json
from pathlib import Path


def equivalent(before, after, key=''):
    if key in ('time', 'deadline') and isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return abs(before-after) <= .0001
    if isinstance(before, dict) and isinstance(after, dict):
        return before.keys() == after.keys() and all(equivalent(before[k], after[k], k) for k in before)
    if isinstance(before, list) and isinstance(after, list):
        return len(before) == len(after) and all(equivalent(a,b,key) for a,b in zip(before,after))
    return before == after


def compare(directory):
    def load(name):
        return json.loads((directory/name).read_text(encoding='utf-8'))
    old, new = load('baseline-ideal.json'), load('ideal.json')
    protected = {key: equivalent(old[key],new[key]) for key in ('heads','actions','scheduled','pending_at_clip_end')}
    old, new = load('baseline-hold-contract.json'), load('hold-contract-after.json')
    protected['hold_contract'] = all(equivalent(old[k],new[k]) for k in ('tests','records','errors'))
    old, new = load('baseline-taps-bench.json'), load('candidate-taps-bench.json')
    delta = new['milliseconds']['p95']-old['milliseconds']['p95']
    old_loop, new_loop = load('baseline-loop.json'), load('loop-final.json')
    order = lambda report: [(head['lane'],head['gesture']) for head in report['heads']]
    deltas = [(b['time']-a['time'])*1000. for a,b in zip(old_loop['heads'],new_loop['heads'])]
    deadline_deltas = [(b['deadline']-a['deadline'])*1000.
                       for a,b in zip(old_loop['scheduled'],new_loop['scheduled'])]
    action_shape = lambda report: [{key:value for key,value in action.items() if key != 'time'}
                                   for action in report['actions']]
    loop = {'heads_baseline':len(old_loop['heads']), 'heads_candidate':len(new_loop['heads']),
            'order_equal':order(old_loop) == order(new_loop),
            'action_kind_contact_coordinate_equal':action_shape(old_loop) == action_shape(new_loop),
            'duplicates_candidate':len(new_loop['heads'])-len({h['track'] for h in new_loop['heads']}),
            'baseline_annotations':old_loop.get('annotation_result'),
            'candidate_annotations':new_loop.get('annotation_result'),
            'input_time_delta_ms_by_ordinal':deltas,
            'max_abs_input_time_delta_ms':max(map(abs,deltas),default=0.),
            'initial_deadline_delta_ms_by_ordinal':deadline_deltas,
            'max_abs_initial_deadline_delta_ms':max(map(abs,deadline_deltas),default=0.),
            'note':'Delay-injected loop exercises the intentionally changed post-capture scheduling path; time differences are reported, not game Bad/Miss.'}
    loop_ok = (loop['order_equal'] and loop['duplicates_candidate'] == 0
               and not new_loop['annotation_result']['missing'] and not new_loop['annotation_result']['extra'])
    return {'protected_contracts': protected, 'timing_tolerance_ms': .1,
            'tap_benchmark_p95_increment_ms': round(delta,3), 'overhead_limit_ms': 2,
            'production_loop':loop, 'passed': all(protected.values()) and delta <= 2 and loop_ok,
            'live_fc_acceptance': 'pending: Support counts as Miss; offline results are not game judgements'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args = parser.parse_args()
    result = compare(args.directory)
    (args.directory/'comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
