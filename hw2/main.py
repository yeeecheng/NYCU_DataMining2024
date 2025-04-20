import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# --- hyper parameter ---
BATCH_SIZE = 256
EPOCHS = 1000
LR = 1e-3
SAVE_PATH = 'ae_weights.pth'
TRAIN_MODE = True

# --- load data ---
train_df = pd.read_csv('./data/training.csv')
test_df = pd.read_csv('./data/test_X.csv')
X_train = train_df.drop(columns=["lettr"]).values.astype(np.float32)
X_test = test_df.values.astype(np.float32)

# --- Normalize ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_main, X_val = train_test_split(X_train, test_size=0.01, random_state=42)
train_loader = DataLoader(TensorDataset(torch.tensor(X_train_main)), batch_size=BATCH_SIZE, shuffle=True)
val_tensor = torch.tensor(X_val).cuda()
test_tensor = torch.tensor(X_test).cuda()

# --- Robust Loss ---
def robust_loss(x, y):
    diff = x - y
    return torch.mean(torch.log(1 + (diff ** 2) / 0.1))

class DeepAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(16, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),

        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )

    def forward(self, x):
        z = self.encoder(x)
        z = F.normalize(z, dim=1)  # L2-normalization
        return self.decoder(z)

model = DeepAutoEncoder().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=30, verbose=True
)

if TRAIN_MODE:
    print("Training...")
    train_loss_list = []
    val_loss_list = []
    best_val_loss = float('inf')
    patience = 100
    patience_counter = 0

    model.train()
    for epoch in range(EPOCHS):
        total_train_loss = 0
        for (x,) in train_loader:
            x = x.cuda()
            optimizer.zero_grad()
            output = model(x)
            loss = robust_loss(output, x)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_recon = model(val_tensor)
            val_loss = robust_loss(val_recon, val_tensor).item()
        model.train()

        train_loss_list.append(avg_train_loss)
        val_loss_list.append(val_loss)
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print("  ↳ Model improved. Weights saved.")
        else:
            patience_counter += 1
            print(f"  ↳ No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
        scheduler.step(val_loss)
else:
    print("Loading weights...")
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()

# --- Loss Curve ---
if TRAIN_MODE:
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label="Train Loss", marker='o')
    plt.plot(range(1, len(val_loss_list) + 1), val_loss_list, label="Val Loss", marker='x')
    plt.title("Training & Validation Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_validation_loss_curve.png")
    plt.show()

# --- inference ---
model.load_state_dict(torch.load(SAVE_PATH))
model.eval()
with torch.no_grad():
    recon = model(test_tensor).cpu().numpy()
    latent = F.normalize(model.encoder(test_tensor), dim=1).cpu().numpy()
    mse = np.mean((recon - X_test) ** 2, axis=1)

# --- save submission.csv ---
submission = pd.DataFrame({'id': np.arange(len(mse)), 'outliers': mse})
submission.to_csv('submission.csv', index=False)
print("Saved submission.csv ")

# --- visualize：PCA + t-SNE ---
latent_norm = StandardScaler().fit_transform(latent)

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(latent_norm)
plt.figure(figsize=(6, 5))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=mse, cmap='coolwarm', s=10)
plt.colorbar(label="Reconstruction Error (MSE)")
plt.title("PCA of AE Latent")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("pca_latent.png")
plt.show()

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(latent_norm)
plt.figure(figsize=(6, 5))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=mse, cmap='coolwarm', s=10)
plt.colorbar(label="Reconstruction Error (MSE)")
plt.title("t-SNE of AE Latent")
plt.xlabel("t-SNE1")
plt.ylabel("t-SNE2")
plt.tight_layout()
plt.savefig("tsne_latent.png")
plt.show()
