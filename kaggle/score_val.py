"""Print accuracy and confusion tables for a zero-shot run (works on the raw output of run_zero_shot.py).

  python score_val.py /kaggle/working/zeroshot_Qwen3-VL-4B-Instruct_v2_probs.csv
Prompts_v2 files also report how often the two option orders gave the same answer and the accuracy of each order alone.
"""
import sys

import pandas as pd

ORDER = ["normal", "stuck_item", "gate_open", "spray", "blocked_lens", "invalid"]
df = pd.read_csv(sys.argv[1])
df["true"] = df.clip_id.str.rsplit("_", n=2).str[0]
for (split, inp, cond), g in df.groupby(["split", "input", "condition"]):
    line = f"\n=== {split} / {inp} / {cond}  n={len(g)}  accuracy={(g.pred == g.true).mean():.3f}"
    if "pred_o1" in g:
        line += f"  | order1 {(g.pred_o1 == g.true).mean():.3f}  order2 {(g.pred_o2 == g.true).mean():.3f}  orders agree {g.agree.mean():.2f}"
    print(line)
    print(pd.crosstab(g.true, g.pred).reindex(columns=ORDER, fill_value=0))
