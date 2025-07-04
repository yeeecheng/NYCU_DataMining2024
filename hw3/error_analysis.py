import pandas as pd

# === Load your data ===
submission_df = pd.read_csv("submission.csv")
behaviors_df = pd.read_csv("./train/train_behaviors.tsv", sep='\t', header=None,
                           names=["impression_id", "user_id", "time", "history", "impressions"])
news_df = pd.read_csv("./train/train_news.tsv", sep='\t', header=None,
                      names=["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"])

# === Preprocess ===
submission_df.rename(columns={'id': 'impression_id'}, inplace=True)
submission_df["impression_id"] = submission_df["impression_id"].astype(str)
behaviors_df["impression_id"] = behaviors_df["impression_id"].astype(str)

for i in range(1, 16):
    submission_df[f'p{i}'] = submission_df[f'p{i}'].astype(float)

merged_df = pd.merge(submission_df, behaviors_df, on="impression_id", how="inner")

# === Collect all false positives ===
false_positives = []

for idx, row in merged_df.iterrows():
    impressions = row['impressions'].split()
    news_ids = [x.split('-')[0] for x in impressions]
    labels = [int(x.split('-')[1]) for x in impressions]
    scores = [row[f'p{i+1}'] for i in range(len(impressions))]

    for nid, score, label in zip(news_ids, scores, labels):
        if label == 0 and score > 0.9:
            false_positives.append((nid, score))

# === Group and count most frequently mispredicted news ===
fp_df = pd.DataFrame(false_positives, columns=["news_id", "score"])
fp_counts = fp_df["news_id"].value_counts().reset_index()
fp_counts.columns = ["news_id", "false_positive_count"]

# === Merge with news_df to get categories and titles ===
fp_summary = pd.merge(fp_counts, news_df[["news_id", "category", "title"]], on="news_id", how="left")

# === 顯示最常被誤判的前幾篇新聞 ===
import pandas as pd
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt

# 排名前 10 的 false positive 新聞
top_fp = fp_summary.head(10)
display(top_fp)

# 類別統計
category_fp = fp_summary["category"].value_counts().reset_index()
category_fp.columns = ["category", "false_positive_count"]
print("\n[Top Misleading Categories (False Positives)]")
print(category_fp.head(10))
