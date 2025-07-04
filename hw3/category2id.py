import json
import pandas as pd

news_df = pd.read_csv("train/train_news.tsv", sep="\t", header=None,
                      names=["id", "category", "subcategory", "title", "abstract", "title_entities", "abstract_entities"])

categories = sorted(news_df['category'].dropna().unique())
category2id = {cat: i for i, cat in enumerate(categories)}

with open("train/category2id.json", "w") as f:
    json.dump(category2id, f, indent=2)

print(f"✅ Saved category2id.json with {len(category2id)} categories")
