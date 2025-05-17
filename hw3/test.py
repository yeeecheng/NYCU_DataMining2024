import torch
import csv
import os
import argparse
import pandas as pd
from transformers import BertTokenizer
from tqdm import tqdm
from main import DSSMModel, build_vocab_and_encode  # 假設定義都在 main.py 中

@torch.no_grad()
def predict_submission(model, news_map, test_behaviors_path, max_len, device, save_path='submission.csv'):
    model.eval()
    results = []

    with open(test_behaviors_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in tqdm(reader):
            uid = row['id']
            clicks = row['clicked_news'].split() if row['clicked_news'] else []
            impressions = row['impressions'].split()
            news_ids = impressions

            if not news_ids:
                continue

            click_vecs = [news_map[cid] for cid in clicks if cid in news_map]
            if not click_vecs:
                click_vecs = [torch.zeros(max_len, dtype=torch.long)]  # fallback for empty history
            click_tensor = torch.stack(click_vecs).unsqueeze(0).to(device)  # [1, K, T]

            cand_vecs = [news_map[nid] if nid in news_map else torch.zeros(max_len, dtype=torch.long) for nid in news_ids]
            cand_tensor = torch.stack(cand_vecs).to(device)  # [15, T]

            click_tensor = click_tensor.expand(cand_tensor.size(0), -1, -1)  # [15, K, T]
            scores = model(click_tensor, cand_tensor).sigmoid().cpu().numpy()

            assert len(scores) == 15, f"Expect 15 scores but got {len(scores)}"
            results.append([uid] + scores.tolist())

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id'] + [f'p{i+1}' for i in range(15)])
        for row in results:
            writer.writerow(row)
    print(f"✅ Submission saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--news_path', type=str, default='./test/test_news.tsv')
    parser.add_argument('--test_behaviors_path', type=str, default='./test/test_behaviors.tsv')
    parser.add_argument('--model_path', type=str, default='dssm_model.pth')
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--save_path', type=str, default='submission.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    news_df = pd.read_csv(args.news_path, sep='\t', header=0)
    news_map, _ = build_vocab_and_encode(news_df, tokenizer, max_len=args.max_len)

    model = DSSMModel(vocab_size=tokenizer.vocab_size, emb_dim=args.emb_dim)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    predict_submission(model, news_map, args.test_behaviors_path, args.max_len, device, save_path=args.save_path)
