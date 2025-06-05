# DSSM Training + Ranking Loss + Full Feature Encoding (title + abstract + category + entities + entity_emb)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import BertTokenizer
from collections import Counter
import random
import os
import json
from tqdm import tqdm

# ==== 1. Model Definition ====
class NewsEncoder(nn.Module):
    def __init__(self, vocab_size, cat_size, ent_size, emb_dim=128):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size + 1000, emb_dim, padding_idx=0)
        self.cat_emb = nn.Embedding(cat_size + 1, emb_dim)
        self.ent_emb = nn.Embedding(ent_size + 1, emb_dim)
        self.linear = nn.Linear(emb_dim * 3, emb_dim)

    def forward(self, news_input, cat_input, ent_input):
        if news_input.dim() == 2:
            word_vec = self.word_emb(news_input)
            mask = (news_input != 0).unsqueeze(-1)
            word_vec = (word_vec * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            cat_vec = self.cat_emb(cat_input)
            ent_vec = self.ent_emb(ent_input)
        elif news_input.dim() == 3:
            B, K, T = news_input.size()
            news_input_flat = news_input.reshape(B * K, T)
            word_vec = self.word_emb(news_input_flat)
            mask = (news_input_flat != 0).unsqueeze(-1)
            word_vec = (word_vec * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            word_vec = word_vec.view(B, K, -1)
            cat_vec = self.cat_emb(cat_input)
            ent_vec = self.ent_emb(ent_input)
        else:
            raise ValueError("Invalid shape")

        fused = torch.cat([word_vec, cat_vec, ent_vec], dim=-1)
        return F.relu(self.linear(fused))

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
    def __init__(self, vocab_size, cat_size, ent_size, emb_dim=128):
        super().__init__()
        self.news_encoder = NewsEncoder(vocab_size, cat_size, ent_size, emb_dim)
        self.user_encoder = UserEncoder(emb_dim)

    def forward(self, clicks, pos, neg, click_cats, pos_cat, neg_cat, click_ents, pos_ent, neg_ent):
        user_vec = self.user_encoder(self.news_encoder(clicks, click_cats, click_ents))
        pos_vec = self.news_encoder(pos, pos_cat, pos_ent)
        neg_vec = self.news_encoder(neg, neg_cat, neg_ent)
        return (user_vec * pos_vec).sum(dim=-1), (user_vec * neg_vec).sum(dim=-1)

    def predict(self, clicks, cands, click_cats, cand_cats, click_ents, cand_ents):
        user_vec = self.user_encoder(self.news_encoder(clicks, click_cats, click_ents))
        cand_vec = self.news_encoder(cands, cand_cats, cand_ents)
        return (user_vec * cand_vec).sum(dim=-1)

# ==== 2. Preprocessing ====
def build_vocab_and_encode(news_df, tokenizer, max_len=80):
    news_map = {}
    cat_set = set()
    ent_set = set()
    news_df = news_df.dropna(subset=['title'])

    for _, row in news_df.iterrows():
        text = f"{row['title']} {row['abstract'] or ''}"
        entities = []

        try:
            ents = json.loads(row['title_entities']) + json.loads(row['abstract_entities'])
            keywords = ' '.join(e['SurfaceForms'][0] for e in ents if 'SurfaceForms' in e and e['SurfaceForms'])
            text += ' ' + keywords
            entities = [e['WikidataId'] for e in ents if 'WikidataId' in e]
        except:
            pass

        tokens = tokenizer(text, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        cat = f"{row['category']}_{row['subcategory']}"
        cat_set.add(cat)
        news_map[row['news_id']] = (tokens['input_ids'].squeeze(0), cat, entities)

    cat2id = {c: i+1 for i, c in enumerate(sorted(cat_set))}
    all_ent_ids = {e for _, _, ents in news_map.values() for e in ents}
    ent2id = {e: i+1 for i, e in enumerate(sorted(all_ent_ids))}
    return news_map, cat2id, ent2id

# ==== 3. Triplet Building ====
def build_triplets(behaviors_df, news_map, cat2id, ent2id, max_clicks=10):
    triplets = []
    for _, row in behaviors_df.iterrows():
        clicks = str(row['clicked_news']).split()[-max_clicks:]
        impressions = str(row['impressions']).split()
        pos_ids = [i.split('-')[0] for i in impressions if i.endswith('-1') and i.split('-')[0] in news_map]
        neg_ids = [i.split('-')[0] for i in impressions if i.endswith('-0') and i.split('-')[0] in news_map]
        if not pos_ids or not neg_ids:
            continue
        for pos_id in pos_ids:
            for _ in range(2):
                neg_id = random.choice(neg_ids)
                click_vecs, click_cats, click_ents = [], [], []
                for cid in clicks:
                    if cid in news_map:
                        t, c, eids = news_map[cid]
                        click_vecs.append(t)
                        click_cats.append(cat2id.get(c, 0))
                        ent_id = ent2id.get(eids[0], 0) if eids else 0
                        click_ents.append(ent_id)
                if not click_vecs:
                    continue
                click_tensor = torch.stack(click_vecs)
                click_cat_tensor = torch.tensor(click_cats)
                click_ent_tensor = torch.tensor(click_ents)

                pos_tensor, pos_cat, pos_ents = news_map[pos_id]
                neg_tensor, neg_cat, neg_ents = news_map[neg_id]
                pos_ent = torch.tensor(ent2id.get(pos_ents[0], 0) if pos_ents else 0)
                neg_ent = torch.tensor(ent2id.get(neg_ents[0], 0) if neg_ents else 0)

                triplets.append((click_tensor, pos_tensor, neg_tensor,
                                 click_cat_tensor, torch.tensor(cat2id.get(pos_cat, 0)), torch.tensor(cat2id.get(neg_cat, 0)),
                                 click_ent_tensor, pos_ent, neg_ent))
    return triplets

# ==== 4. Collate ====
def collate_fn(batch):
    clicks, pos, neg, click_cats, pos_cats, neg_cats, click_ents, pos_ents, neg_ents = zip(*batch)
    clicks = nn.utils.rnn.pad_sequence(clicks, batch_first=True)
    click_cats = nn.utils.rnn.pad_sequence(click_cats, batch_first=True)
    click_ents = nn.utils.rnn.pad_sequence(click_ents, batch_first=True)
    return clicks, torch.stack(pos), torch.stack(neg), click_cats, torch.stack(pos_cats), torch.stack(neg_cats), click_ents, torch.stack(pos_ents), torch.stack(neg_ents)

# Dataset for triplets. Defined at top level so it can be pickled by
# DataLoader workers when `num_workers > 0`.
class TripletSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ==== 5. Training ====
def train_one_epoch(model, loader, optimizer, margin, device):
    model.train()
    total_loss = 0
    for clicks, pos, neg, click_cats, pos_cats, neg_cats, click_ents, pos_ents, neg_ents in tqdm(loader):
        clicks = clicks.to(device)
        pos = pos.to(device)
        neg = neg.to(device)
        click_cats = click_cats.to(device)
        pos_cats = pos_cats.to(device)
        neg_cats = neg_cats.to(device)
        click_ents = click_ents.to(device)
        pos_ents = pos_ents.to(device)
        neg_ents = neg_ents.to(device)

        optimizer.zero_grad()
        pos_score, neg_score = model(clicks, pos, neg, click_cats, pos_cats, neg_cats, click_ents, pos_ents, neg_ents)
        loss = F.margin_ranking_loss(pos_score, neg_score, torch.ones_like(pos_score), margin=margin)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, margin, device):
    """Evaluate on validation set using the same ranking loss."""
    model.eval()
    total_loss = 0
    for clicks, pos, neg, click_cats, pos_cats, neg_cats, click_ents, pos_ents, neg_ents in loader:
        clicks = clicks.to(device)
        pos = pos.to(device)
        neg = neg.to(device)
        click_cats = click_cats.to(device)
        pos_cats = pos_cats.to(device)
        neg_cats = neg_cats.to(device)
        click_ents = click_ents.to(device)
        pos_ents = pos_ents.to(device)
        neg_ents = neg_ents.to(device)

        pos_score, neg_score = model(clicks, pos, neg, click_cats, pos_cats, neg_cats, click_ents, pos_ents, neg_ents)
        loss = F.margin_ranking_loss(pos_score, neg_score, torch.ones_like(pos_score), margin=margin)
        total_loss += loss.item()
    return total_loss / len(loader)

# ==== 6. Main Entry ====
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--news_path', type=str, default='./train/train_news.tsv')
    parser.add_argument('--behaviors_path', type=str, default='./train/train_behaviors.tsv')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--max_len', type=int, default=80)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--save_model', type=str, default='dssm_full_entity.pth')
    parser.add_argument('--val_split', type=float, default=0.1, help='Fraction of data used for validation')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    news_df = pd.read_csv(args.news_path, sep='\t', header=0)
    behaviors_df = pd.read_csv(args.behaviors_path, sep='\t', header=0)

    print("[INFO] Encoding news...")
    news_map, cat2id, ent2id = build_vocab_and_encode(news_df, tokenizer, max_len=args.max_len)
    print("[INFO] Building triplets...")
    triplets = build_triplets(behaviors_df, news_map, cat2id, ent2id)

    dataset = TripletSet(triplets)
    if args.val_split > 0:
        train_len = int(len(dataset) * (1 - args.val_split))
        val_len = len(dataset) - train_len
        train_set, val_set = random_split(dataset, [train_len, val_len])
    else:
        train_set, val_set = dataset, None

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = None if val_set is None else DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = DSSMModel(tokenizer.vocab_size, len(cat2id), len(ent2id), emb_dim=args.emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, args.margin, device)
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, args.margin, device)
        else:
            val_loss = train_loss
        print(f"Epoch {epoch+1}/{args.epochs} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            print(f"✅ New best model found! Saving...")
            torch.save(model.state_dict(), str(epoch) + "_" + args.save_model)

    import pickle
    # ✅ Save category and entity mappings
    with open('cat2id.pkl', 'wb') as f:
        pickle.dump(cat2id, f)
    with open('ent2id.pkl', 'wb') as f:
        pickle.dump(ent2id, f)
    print("✅ cat2id and ent2id mappings saved.")
