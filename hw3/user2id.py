import json
import pandas as pd

user_ids = set()

with open("train/train_behaviors.tsv", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i == 0:
            continue
        parts = line.strip().split('\t')
        if len(parts) != 5:
            continue
        _, user_id, _, _, _ = parts
        user_ids.add(user_id)

user_ids = sorted(user_ids)
user2id = {uid: i for i, uid in enumerate(user_ids)}

with open("train/user2id.json", "w") as f:
    json.dump(user2id, f, indent=2)

print(f"✅ Saved user2id.json with {len(user2id)} users")
