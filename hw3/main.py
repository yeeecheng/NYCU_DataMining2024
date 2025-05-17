# DSSM Training + Debug-Fixed Pipeline

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import BertTokenizer
from collections import Counter
import random
import os
import pickle
from tqdm import tqdm

# ==== 1. Model Definition ====
class NewsEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1000, emb_dim, padding_idx=0)
        self.linear = nn.Linear(emb_dim, emb_dim)

    def forward(self, news_input):
        if news_input.dim() == 2:
            emb = self.embedding(news_input)
            mask = (news_input != 0).unsqueeze(-1)
            vec = (emb * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        elif news_input.dim() == 3:
            emb = self.embedding(news_input)
            mask = (news_input != 0).unsqueeze(-1)
            vec = (emb * mask).sum(dim=2) / (mask.sum(dim=2) + 1e-8)
        else:
            raise ValueError("Invalid input shape for news_input")
        return F.relu(self.linear(vec))

class UserEncoder(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_dim, num_heads=4, batch_first=True)
        self.linear = nn.Linear(emb_dim, emb_dim)

    def forward(self, clicked_vecs):
        attn_output, _ = self.attn(clicked_vecs, clicked_vecs, clicked_vecs)
        user_vec = attn_output.mean(dim=1)
        return F.relu(self.linear(user_vec))

class DSSMModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=128):
        super().__init__()
        self.news_encoder = NewsEncoder(vocab_size, emb_dim)
        self.user_encoder = UserEncoder(emb_dim)

    def forward(self, clicked_news, candidate_news):
        clicked_vecs = self.news_encoder(clicked_news)
        user_vec = self.user_encoder(clicked_vecs)
        candidate_vec = self.news_encoder(candidate_news)
        score = (user_vec * candidate_vec).sum(dim=-1)
        return score

# ==== 2. Dataset Loader ====
class NewsDataset(Dataset):
    def __init__(self, clicked_news, candidate_news, labels):
        self.clicked_news = clicked_news
        self.candidate_news = candidate_news
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.clicked_news[idx], self.candidate_news[idx], self.labels[idx]

# ==== 3. Training Function ====
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for clicks, cands, labels in tqdm(dataloader, desc="Training"):
        clicks = clicks.to(device)
        cands = cands.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()
        scores = model(clicks, cands)
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# ==== 4. Evaluation Function ====
def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for clicks, cands, labels in dataloader:
            clicks = clicks.to(device)
            cands = cands.to(device)
            labels = labels.to(device).float()
            scores = model(clicks, cands)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(scores.cpu().numpy())
    return roc_auc_score(all_labels, all_preds)

# ==== 6. Text Preprocessing ====
def build_vocab_and_encode(news_df, tokenizer, max_len=20):
    news_id_to_tensor = {}
    news_df = news_df[news_df['title'].notna() & (news_df['title'].str.strip() != '')]
    for _, row in news_df.iterrows():
        tokens = tokenizer(row['title'], padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        news_id_to_tensor[row['news_id']] = tokens['input_ids'].squeeze(0)
    return news_id_to_tensor, None

def compute_news_vectors(news_map, model, device):
    model.eval()
    vec_map = {}
    with torch.no_grad():
        for nid, ids in news_map.items():
            vec = model.news_encoder(ids.unsqueeze(0).to(device)).squeeze(0).cpu()
            vec_map[nid] = vec
    return vec_map

# ==== 7. Negative Sampling with Downsampling ====
def build_training_samples(behaviors_df, news_map, max_clicks=10, cache_path=None):

    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    clicked_list, candidate_list, labels = [], [], []
    for _, row in behaviors_df.iterrows():
        clicks = str(row['clicked_news']).split()[-max_clicks:]
        impressions = str(row['impressions']).split()
        for imp in impressions:
            if '-' not in imp:
                continue
            parts = imp.split('-')
            if len(parts) != 2:
                continue
            news_id, clicked = parts
            if news_id not in news_map:
                continue
            if int(clicked) == 0 and random.random() > 0.25:
                continue  # downsample negative samples
            click_vecs = [news_map[cid] for cid in clicks if cid in news_map]
            if not click_vecs:
                click_vecs = [torch.zeros_like(news_map[news_id])]
            click_tensor = torch.stack(click_vecs)
            candidate_tensor = news_map[news_id]
            clicked_list.append(click_tensor)
            candidate_list.append(candidate_tensor)
            labels.append(int(clicked))
    print("[INFO] Label distribution:", Counter(labels))

    if cache_path:
        with open(cache_path, 'wb') as f:
            pickle.dump((clicked_list, candidate_list, labels), f)

    return clicked_list, candidate_list, labels

# ==== 9. Collate ====
def collate_fn(batch):
    clicks = [b[0] for b in batch]
    cands = [b[1] for b in batch]
    labels = [b[2] for b in batch]
    clicks_padded = nn.utils.rnn.pad_sequence(clicks, batch_first=True)
    cands_tensor = torch.stack(cands)
    labels_tensor = torch.tensor(labels).float()
    return clicks_padded, cands_tensor, labels_tensor

# ==== 10. Main Entry ====
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--news_path', type=str, default='./train/train_news.tsv')
    parser.add_argument('--behaviors_path', type=str, default='./train/train_behaviors.tsv')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--save_model', type=str, default='dssm_model.pth')
    args = parser.parse_args()

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print("[INFO] Loading news and behaviors...")
    news_df = pd.read_csv(args.news_path, sep='\t', names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
    behaviors_df = pd.read_csv(args.behaviors_path, sep='\t', header=0)

    print("[INFO] Building vocab and encoding news...")
    news_map, _ = build_vocab_and_encode(news_df, tokenizer, max_len=args.max_len)

    model = DSSMModel(tokenizer.vocab_size, emb_dim=args.emb_dim).to(device)
    print("[INFO] Precomputing news vectors...")
    news_vecs = compute_news_vectors(news_map, model, device)

    print("[INFO] Building training samples...")
    train_clicked, train_candidates, train_labels = build_training_samples(behaviors_df, news_vecs, cache_path=args.cache_path)
    train_dataset = NewsDataset(train_clicked, train_candidates, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos = sum(train_labels)
    neg = len(train_labels) - pos
    pos_weight = torch.tensor([neg / (pos + 1e-8)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("[INFO] Starting training...")
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f}")

    torch.save(model.state_dict(), args.save_model)
    print(f"[INFO] Model saved to {args.save_model}")
