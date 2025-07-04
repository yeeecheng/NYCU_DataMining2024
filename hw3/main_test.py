import torch
import csv
import os
import argparse
import pandas as pd
import pickle
from transformers import BertTokenizer
from tqdm import tqdm
from main import DSSMModel
import json

def build_vocab_and_encode(news_df, tokenizer, cat2id, ent2id, max_len=80):
    news_map = {}
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
        cat_id = cat2id.get(cat, 0)
        ent_id = ent2id.get(entities[0], 0) if entities else 0

        news_map[row['news_id']] = (tokens['input_ids'].squeeze(0), cat_id, ent_id)
    return news_map


@torch.no_grad()
def predict_submission(model, news_map, cat2id, ent2id, test_behaviors_path, max_len, device, save_path='submission.csv'):
    model.eval()
    results = []

    with open(test_behaviors_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in tqdm(reader):
            uid = row['id']
            clicks = row['clicked_news'].split() if row['clicked_news'] else []
            impressions = row['impressions'].split()
            news_ids = [imp.split('-')[0] for imp in impressions]

            if not news_ids:
                continue

            click_vecs, click_cats, click_ents = [], [], []
            for cid in clicks:
                if cid in news_map:
                    ids, cat_id, ent_id = news_map[cid]
                    click_vecs.append(ids)
                    click_cats.append(cat_id)
                    click_ents.append(ent_id)
            if not click_vecs:
                click_vecs = [torch.zeros(max_len, dtype=torch.long)]
                click_cats = [0]
                click_ents = [0]

            click_tensor = torch.stack(click_vecs).unsqueeze(0).expand(len(news_ids), -1, -1).to(device)
            click_cat_tensor = torch.tensor(click_cats).unsqueeze(0).expand(len(news_ids), -1).to(device)
            click_ent_tensor = torch.tensor(click_ents).unsqueeze(0).expand(len(news_ids), -1).to(device)

            cand_vecs, cand_cats, cand_ents = [], [], []
            for nid in news_ids:
                if nid in news_map:
                    ids, cat_id, ent_id = news_map[nid]
                else:
                    ids = torch.zeros(max_len, dtype=torch.long)
                    cat_id = 0
                    ent_id = 0
                cand_vecs.append(ids)
                cand_cats.append(cat_id)
                cand_ents.append(ent_id)

            cand_tensor = torch.stack(cand_vecs).to(device)
            cand_cat_tensor = torch.tensor(cand_cats).to(device)
            cand_ent_tensor = torch.tensor(cand_ents).to(device)

            scores = model.predict(click_tensor, cand_tensor, click_cat_tensor, cand_cat_tensor, click_ent_tensor, cand_ent_tensor)
            scores = scores.sigmoid().cpu().numpy()

            results.append([uid] + scores.tolist())

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id'] + [f'p{i+1}' for i in range(15)])
        for row in results:
            writer.writerow(row)
    print(f"✅ Submission saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--news_path', type=str, default='./train/train_news.tsv')
    parser.add_argument('--test_behaviors_path', type=str, default='./train/train_behaviors.tsv')
    parser.add_argument('--model_path', type=str, default='1_dssm_full_entity.pth')
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=40)
    parser.add_argument('--save_path', type=str, default='submission.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # ✅ Load category and entity mappings from training
    with open('cat2id.pkl', 'rb') as f:
        cat2id = pickle.load(f)
    with open('ent2id.pkl', 'rb') as f:
        ent2id = pickle.load(f)

    print("[INFO] Loading and encoding news...")
    news_df = pd.read_csv(args.news_path, sep='\t', header=0)
    news_map = build_vocab_and_encode(news_df, tokenizer, cat2id=cat2id, ent2id=ent2id, max_len=args.max_len)

    model = DSSMModel(vocab_size=tokenizer.vocab_size, cat_size=len(cat2id), ent_size=len(ent2id), emb_dim=args.emb_dim)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    predict_submission(model, news_map, cat2id, ent2id, args.test_behaviors_path, args.max_len, device, save_path=args.save_path)