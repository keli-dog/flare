import glob
import pickle
import re

eps = set()
files = sorted(glob.glob("results/analyze_recs/tests_seen_*_flare_paper.p"))
for p in files:
    recs = pickle.load(open(p, "rb"))
    for r in recs:
        ep = r.get("number_of_this_episode")
        if ep is not None:
            eps.add(int(ep))
    m = re.search(r"from_(\d+)_to_(\d+)", p)
    span = f"[{m.group(1)},{m.group(2)})" if m else p
    print(f"{p.split('/')[-1]}: {len(recs)} records, span {span}")

missing = [i for i in range(1533) if i not in eps]
print(f"\nunique={len(eps)}/1533 missing={len(missing)}")
if missing:
    print(f"missing range {missing[0]}..{missing[-1]}")
    print(f"first 20 missing: {missing[:20]}")
    print(f"last 20 missing: {missing[-20:]}")
    for a, b in [(0, 204), (204, 537), (537, 870), (870, 1203), (1203, 1533)]:
        m = [x for x in missing if a <= x < b]
        print(f"  [{a},{b}): {len(m)} missing")
