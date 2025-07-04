# NRMS Inference Code with Kaggle Submission Format
# Compatible with training pipeline and Kaggle evaluation requirements

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import argparse
import os
import json
import csv
from tqdm import tqdm

from train_nrms_MIND import NRMS, ClickDataset
# from train_dataset import

# ========== Inference Function ==========
@torch.no_grad()
def run_submission(model, dataloader, device, save_path='submission.csv'):
    model.eval()
    results = []

    # Read full behavior file into list to ensure alignment
    with open(dataloader.dataset.samples_path, 'r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f, delimiter='\t'))

    for i, batch in enumerate(dataloader):
        row = reader[i]
        sid = row['id']
        imps = row['impressions'].split()
        news_ids = [imp.split('-')[0] for imp in imps]

        sid_tensor, user_id, clicks, cands, click_cats, cand_cats, _ = batch

        user_id = user_id.to(device)
        clicks = clicks.to(device)
        cands = cands.to(device)
        click_cats = click_cats.to(device)
        cand_cats = cand_cats.to(device)

        if clicks.dim() == 1:
            clicks = clicks.unsqueeze(0)
        if cands.dim() == 1:
            cands = cands.unsqueeze(0)

        scores = model(user_id, clicks, click_cats, cands, cand_cats)
        scores = scores.squeeze(0).tolist()

        row_result = [sid] + scores
        results.append(row_result)

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id'] + [f'p{i+1}' for i in range(15)])
        for row in results:
            writer.writerow(row)
    print(f"✅ Submission saved to {save_path}")

# ========== Main ==========
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--behaviors', type=str, default='test/test_behaviors.tsv')
    parser.add_argument('--news', type=str, default='test/test_news.tsv')
    parser.add_argument('--save_dir', type=str, default='nrms_ckpt')
    parser.add_argument('--model_name', type=str, default='best_model_epoch1.pth')
    parser.add_argument('--submission_name', type=str, default='submission.csv')
    parser.add_argument('--category2id', type=str, default='train/category2id.json')
    parser.add_argument('--user2id', type=str, default='train/user2id.json')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    config = {
        'freeze_embedding': False,
        'word_embedding_dim': 200,
        'dropout_prob': 0.1,
        'num_words_title': 20,
        'num_attention_heads': 16,
        'news_dim': 128,
        'news_query_vector_dim': 64,
        'user_query_vector_dim': 64,
        'user_log_length': 20,
        'save_dir': args.save_dir,
        'num_categories': None,
        'num_users': None,
        'pos_weight': 5.0,
        'vocab_size': 1,
        'category2id': args.category2id,
        'user2id': args.user2id
    }

    test_dataset = ClickDataset(args.behaviors, args.news, config, mode='test')
    test_dataset.samples_path = args.behaviors

    model = NRMS(config).to(device)

    model_path = os.path.join(args.save_dir, args.model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Loaded model from {model_path}")

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    run_submission(model, test_loader, device, save_path=args.submission_name)
