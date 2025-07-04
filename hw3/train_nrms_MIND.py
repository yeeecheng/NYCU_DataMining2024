# NRMS Pipeline v2 - Fixed Version (No embedding_matrix, runnable)
# ✅ Fix: Ensure vocab_size = number of unique news_ids (entity2id) + 1

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import argparse
import json
from tqdm import tqdm

from layers import MultiHeadSelfAttention, AttentionPooling

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='nrms_ckpt')
    parser.add_argument('--category2id', type=str, default='train/category2id.json')
    parser.add_argument('--user2id', type=str, default='train/user2id.json')
    parser.add_argument('--behaviors', type=str, default='train/train_behaviors.tsv')
    parser.add_argument('--news', type=str, default='train/train_news.tsv')
    return parser.parse_args()

args = parse_args()

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


class ClickDataset(Dataset):
    def __init__(self, behaviors_path, news_path, config, mode='train'):
        self.samples = []
        self.config = config
        self.mode = mode

        with open(config['category2id']) as f:
            self.cat2id = json.load(f)
            self.cat2id['<UNK>'] = len(self.cat2id)

        with open(config['user2id']) as f:
            self.user2id = json.load(f)
            self.user2id['<UNK>'] = len(self.user2id)

        news_df = pd.read_csv(news_path, sep='\t', names=[
            'news_id', 'category', 'subcategory', 'title', 'abstract', 'URL', 'title_entities', 'abstract_entities'])

        # ========= entity2id loading / saving =========
        if self.mode == 'test':
            with open(os.path.join(config['save_dir'], 'entity2id.json')) as f:
                self.entity2id = json.load(f)
        else:
            self.entity2id = {}
            for row in news_df.itertuples():
                self.entity2id[row.news_id] = len(self.entity2id)
            with open(os.path.join(config['save_dir'], 'entity2id.json'), 'w') as f:
                json.dump(self.entity2id, f)

        config['vocab_size'] = len(self.entity2id) + 1
        config['num_categories'] = len(self.cat2id)
        config['num_users'] = len(self.user2id)

        self.news_dict = {
            nid: (self.entity2id.get(nid, 0), self.cat2id.get(cat, self.cat2id['<UNK>']))
            for nid, cat in zip(news_df.news_id, news_df.category)
        }

        pos_cnt, neg_cnt = 0, 0
        with open(behaviors_path, encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    continue
                parts = line.strip().split('\t')
                if len(parts) != 5:
                    continue
                sid, user_id, _, history, impressions = parts
                user_idx = self.user2id.get(user_id, self.user2id['<UNK>'])
                clicks = history.split() if history else []
                imps = impressions.strip().split()

                if self.mode == 'train':
                    cand = [imp.split('-')[0] for imp in imps]
                    labels = [int(imp.split('-')[1]) for imp in imps]
                    pos_cnt += sum(labels)
                    neg_cnt += len(labels) - sum(labels)
                else:
                    cand = [imp.split('-')[0] if '-' in imp else imp for imp in imps]
                    labels = [0] * len(cand)  # dummy labels

                if not cand:
                    continue
                self.samples.append((sid if self.mode != 'train' else None, user_idx, clicks[:config['user_log_length']], cand, labels))

        if self.mode == 'train':
            self.pos_weight = neg_cnt / (pos_cnt + 1e-6)
        else:
            self.pos_weight = 1.0  # dummy

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, user_id, clicks, candidates, labels = self.samples[idx]
        click_ids = [self.news_dict.get(nid, (0, 0))[0] for nid in clicks]
        click_cats = [self.news_dict.get(nid, (0, 0))[1] for nid in clicks]
        while len(click_ids) < self.config['user_log_length']:
            click_ids.append(0)
            click_cats.append(self.cat2id['<UNK>'])
        cand_ids = [self.news_dict.get(nid, (0, 0))[0] for nid in candidates]
        cand_cats = [self.news_dict.get(nid, (0, 0))[1] for nid in candidates]

        if self.mode == 'train':
            return (
                torch.tensor(user_id),
                torch.tensor(click_ids),
                torch.tensor(cand_ids),
                torch.tensor(click_cats),
                torch.tensor(cand_cats),
                torch.tensor(labels)
            )
        else:
            return (
                sid,
                torch.tensor(user_id),
                torch.tensor(click_ids),
                torch.tensor(cand_ids),
                torch.tensor(click_cats),
                torch.tensor(cand_cats),
                torch.tensor(labels)
            )

def collate_fn(batch):
    user_ids, clicks, cands, click_cats, cand_cats, labels = zip(*batch)
    return (
        torch.stack(user_ids),
        torch.stack(clicks),
        torch.stack(cands),
        torch.stack(click_cats),
        torch.stack(cand_cats),
        torch.stack(labels),
    )

class NewsEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.news_embedding = nn.Embedding(config['vocab_size'], config['news_dim'], padding_idx=0)
        self.cat_embedding = nn.Embedding(config['num_categories'], config['news_dim'])
        self.dropout = nn.Dropout(config['dropout_prob'])
        self.proj = nn.Linear(config['news_dim'] * 2, config['news_dim'])

    def forward(self, news_ids, cat_ids):
        news_vec = self.news_embedding(news_ids)
        cat_vec = self.cat_embedding(cat_ids)
        vec = torch.cat([news_vec, cat_vec], dim=-1)
        return self.proj(self.dropout(vec))

class UserEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config['news_dim']
        self.self_attn = MultiHeadSelfAttention(dim, config['num_attention_heads'], dim // config['num_attention_heads'], dim // config['num_attention_heads'])
        self.attn = AttentionPooling(dim, config['user_query_vector_dim'])
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, clicked_news_vecs, user_embed):
        hist_vec = self.attn(self.self_attn(clicked_news_vecs, clicked_news_vecs, clicked_news_vecs))
        concat = torch.cat([hist_vec, user_embed], dim=-1)
        gate = torch.sigmoid(self.gate(concat))
        return gate * hist_vec + (1 - gate) * user_embed

class NRMS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.news_encoder = NewsEncoder(config)
        self.user_encoder = UserEncoder(config)
        self.user_embedding = nn.Embedding(config['num_users'], config['news_dim'])
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config['pos_weight']]))

    def forward(self, user_ids, clicked_news_ids, click_cats, cand_news_ids, cand_cats, labels=None):
        B, C = clicked_news_ids.size()
        B, K = cand_news_ids.size()
        clicked_vecs = self.news_encoder(clicked_news_ids, click_cats)
        user_embed = self.user_embedding(user_ids)
        user_vec = self.user_encoder(clicked_vecs, user_embed)
        cand_vecs = self.news_encoder(cand_news_ids, cand_cats)
        scores = torch.bmm(cand_vecs, user_vec.unsqueeze(-1)).squeeze(-1)
        if labels is not None:
            loss = self.loss_fn(scores, labels.float())
            return loss, scores
        return scores

if __name__ == '__main__':
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset = ClickDataset(args.behaviors, args.news, config)
    config['pos_weight'] = dataset.pos_weight

    model = NRMS(config).to(device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(config['save_dir'], exist_ok=True)
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for user_ids, clicks, cands, click_cats, cand_cats, labels in tqdm(loader, desc=f"Epoch {epoch+1}"):
            user_ids = user_ids.to(device)
            clicks = clicks.to(device)
            cands = cands.to(device)
            click_cats = click_cats.to(device)
            cand_cats = cand_cats.to(device)
            labels = labels.to(device)
            loss, _ = model(user_ids, clicks, click_cats, cands, cand_cats, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            path = os.path.join(config['save_dir'], f'best_model_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), path)
            print(f"✅ Saved best model to {path}")
