import os, json, re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser



PLANNER_DIR = os.path.dirname(os.path.abspath(__file__))

VALID_ACTIONS = (    'PickupObject', 'PutObject', 'ToggleObject', 'CleanObject',
    'HeatObject', 'CoolObject', 'SliceObject',
)

ACTION_RE = re.compile(
    r'(PickupObject|PutObject|ToggleObject|CleanObject|HeatObject|CoolObject|SliceObject)'
    r'\s*\(\s*([^,)]+?)\s*,\s*([^)]+?)\s*\)',
    re.IGNORECASE,
)


def fix_action_name(action):
    if action.startswith('ickupObject'):
        return 'P' + action
    a = action.strip()
    for valid in VALID_ACTIONS:
        if a.lower() == valid.lower() or a.lower().startswith(valid.lower()):
            return valid
    return a


def normalize_llm_plan_string(raw):
    s = raw.strip()
    # use first non-empty line; drop markdown / numbering prefixes
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^[\d\.\)\-\*]+\s*', '', line)
        s = line
        break
    if not any(s.startswith(a) for a in VALID_ACTIONS):
        s = re.sub(r'^[\W_]+', '', s)
    if s.startswith('ickupObject'):
        s = 'P' + s
    return s


def parse_actions_from_string(s):
    """Parse Action(Obj, Recep) tuples from LLM text."""
    tmp = []
    for action, obj, recep in ACTION_RE.findall(s):
        tmp.append([fix_action_name(action), obj.strip(), recep.strip()])
    if tmp:
        return tmp

    # fallback: legacy comma-split parser
    parts = s.split(',')
    for i in range(len(parts)):
        if 'Object' in parts[i]:
            action = fix_action_name(parts[i].strip())
            obj = parts[i + 1].strip() if i + 1 < len(parts) else "0"
            recep = parts[i + 2].strip() if i + 2 < len(parts) else "0"
            tmp.append([action, obj, recep])
    return tmp


def parse_plan_to_triplet(plan_entry):
    triplet = plan_entry.get('triplet', [])
    if not triplet:
        return []

    first = triplet[0]
    if isinstance(first, list):
        parsed = []
        for item in triplet:
            if isinstance(item, list) and len(item) >= 3:
                action = fix_action_name(str(item[0]))
                if action in VALID_ACTIONS:
                    parsed.append([action, str(item[1]).strip(), str(item[2]).strip()])
                else:
                    # recover from corrupted entries like "PickupObject(Apple"
                    parsed.extend(parse_actions_from_string(",".join(str(x) for x in item)))
            elif isinstance(item, str):
                parsed.extend(parse_actions_from_string(item))
        return parsed

    if isinstance(first, str):
        s = normalize_llm_plan_string(first)
        return parse_actions_from_string(s)

    return []


def main(sp, destination, output_mmp_dir=None):
    save_path = os.path.join(PLANNER_DIR, f'planner_results/{destination}/turbo-bias-{sp}_result.json')
    result = json.load(open(save_path))

    for k in result.keys():
        parsed = parse_plan_to_triplet(result[k])
        result[k]['triplet_beforeDC'] = parsed
        result[k]['triplet'] = parsed

    OPENABLE = ['Fridge', 'Cabinet', 'Microwave', 'Drawer', 'Safe']
    for k in result.keys():
        tmp_actions = []
        tmp_classes = []
        tmp_idxs = []
        skipped_cnt = 0
        DC_indices = []
        for i in range(len(result[k]['triplet_beforeDC'])):
            action = result[k]['triplet_beforeDC'][i][0]
            target = result[k]['triplet_beforeDC'][i][1]
            recp = result[k]['triplet_beforeDC'][i][2]
            if action == 'HeatObject':
                tmp_actions.append('OpenObject')
                tmp_actions.append('PutObject')
                tmp_actions.append('CloseObject')
                tmp_actions.append('ToggleObjectOn')
                tmp_actions.append('ToggleObjectOff')
                tmp_actions.append('OpenObject')
                tmp_actions.append('PickupObject')
                tmp_actions.append('CloseObject')
                
                tmp_classes.append(recp) #Microwave
                tmp_classes.append(recp)
                tmp_classes.append(recp)
                tmp_classes.append(recp)
                tmp_classes.append(recp)
                tmp_classes.append(recp)
                tmp_classes.append(target)
                tmp_classes.append(recp)
                for _ in range(8):
                    tmp_idxs.append(i-skipped_cnt)

                
            elif action == 'CoolObject':
                tmp_actions.append('OpenObject')
                tmp_actions.append('PutObject')
                tmp_actions.append('CloseObject')
                tmp_actions.append('OpenObject')
                tmp_actions.append('PickupObject')
                tmp_actions.append('CloseObject')
                
                tmp_classes.append(recp)
                tmp_classes.append(recp)
                tmp_classes.append(recp)
                tmp_classes.append(recp)
                tmp_classes.append(target)
                tmp_classes.append(recp)
                for _ in range(6):
                    tmp_idxs.append(i-skipped_cnt)


            
            elif action == 'CleanObject':
                tmp_actions.append('PutObject')
                tmp_actions.append('ToggleObjectOn')
                tmp_actions.append('ToggleObjectOff')
                tmp_actions.append('PickupObject')
                
                tmp_classes.append('Sink')
                tmp_classes.append(recp)
                tmp_classes.append(recp)
                tmp_classes.append(target)
                for _ in range(4):
                    tmp_idxs.append(i-skipped_cnt)

            elif action == 'ToggleObject':
                tmp_actions.append('ToggleObjectOn')
                
                tmp_classes.append(target)
                for _ in range(1):
                    tmp_idxs.append(i-skipped_cnt)
                    
            elif action == 'PickupObject':
                if recp in OPENABLE:
                    tmp_actions.append('OpenObject')            
                    tmp_actions.append('PickupObject')
                    tmp_actions.append('CloseObject')            
                    tmp_classes.append(recp)
                    tmp_classes.append(target)
                    tmp_classes.append(recp)
                    for _ in range(3):
                        tmp_idxs.append(i-skipped_cnt)
                
                elif target == 'Laptop':  # Manually postprocess Laptop
                    tmp_actions.append('CloseObject')
                    tmp_actions.append('PickupObject')
                    tmp_classes.append(target)
                    tmp_classes.append(target)
                    for _ in range(2):
                        tmp_idxs.append(i-skipped_cnt)

                else:
                    tmp_actions.append('PickupObject')
                    tmp_classes.append(target)
                    for _ in range(1):
                        tmp_idxs.append(i-skipped_cnt)

            
            elif action == 'SliceObject':
                tmp_actions.append('SliceObject')
                
                tmp_classes.append(target)
                for _ in range(1):
                    tmp_idxs.append(i-skipped_cnt)
                    
            elif action == 'PutObject':
                if recp in OPENABLE:
                    tmp_actions.append('OpenObject')            
                    tmp_actions.append('PutObject')
                    tmp_actions.append('CloseObject')            
                    tmp_classes.append(recp)
                    tmp_classes.append(recp)
                    tmp_classes.append(recp)
                    for _ in range(3):
                        tmp_idxs.append(i-skipped_cnt)
                else:
                    tmp_actions.append('PutObject')
                    tmp_classes.append(recp)
                    for _ in range(1):
                        tmp_idxs.append(i-skipped_cnt)
            else:
                print(action, 'not supported')
                skipped_cnt+=1
            result[k]['low_actions'] = tmp_actions
            result[k]['low_classes'] = tmp_classes
            result[k]['high_idxs'] = tmp_idxs
        for j in reversed(DC_indices):
            del result[k]['triplet'][j]

    with open(save_path, 'w') as f:
        json.dump(result, f, indent=4)

    if output_mmp_dir:
        os.makedirs(output_mmp_dir, exist_ok=True)
        mmp_path = os.path.join(output_mmp_dir, f'{sp}.json')
        with open(mmp_path, 'w') as f:
            json.dump(result, f, indent=4)
        print(f'Wrote {mmp_path}')

if __name__ == "__main__":
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--dn', help='desination', default='qwen3.7-plus', type=str)
    parser.add_argument(
        '--output-mmp-dir',
        help='Also write postprocessed plans as MMP_results/{split}.json',
        default=None,
        type=str,
    )
    parser.add_argument(
        '--split',
        help='Run a single split only (default: all four splits)',
        default=None,
        choices=['tests_seen', 'tests_unseen', 'valid_seen', 'valid_unseen'],
    )
    args = parser.parse_args()

    splits = [args.split] if args.split else ['tests_seen', 'tests_unseen', 'valid_seen', 'valid_unseen']
    for split in splits:
        main(split, args.dn, output_mmp_dir=args.output_mmp_dir)
