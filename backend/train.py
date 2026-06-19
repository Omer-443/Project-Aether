"""backend/train.py - Parallel GPU CDE Training with Signature-Resilient Patch
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# 🚨 THE FINAL SURGICAL RUNTIME PATCH
# This intercepts the call signature at the top-level Dynamo interface 
# and explicitly prevents 'wrapping' from slipping down into the execution frame.
import torch._dynamo

if hasattr(torch._dynamo, 'disable'):
    _orig_dynamo_disable = torch._dynamo.disable

    def _signature_safe_disable(*args, **kwargs):
        # Drop the version-mismatched keyword safely before invoking the tracer
        kwargs.pop('wrapping', None)
        return _orig_dynamo_disable(*args, **kwargs)

    # Apply back to the main access locations scanned by the optimizer compiler hooks
    torch._dynamo.disable = _signature_safe_disable
    
    if hasattr(torch, '_compile') and hasattr(torch._compile, '_disable_dynamo'):
        torch._compile._disable_dynamo = _signature_safe_disable

# Also safe-guard the standard compiler endpoint
import torch.compiler
if hasattr(torch.compiler, 'disable'):
    _orig_compiler_disable = torch.compiler.disable
    def _compiler_safe_disable(*args, **kwargs):
        kwargs.pop('wrapping', None)
        return _orig_compiler_disable(*args, **kwargs)
    torch.compiler.disable = _compiler_safe_disable
# ---------------------------------------------------------

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, confusion_matrix
)
from tqdm import tqdm

try:
    import torchcde
except ImportError:
    raise ImportError("🚨 torchcde missing. Run '!pip install torchcde'")

# --- Kaggle Environment Setup ---
BASE_DIR = Path("/kaggle/input/datasets/omerfaisal443/aether-data/backend")
DATA_DIR = Path("/kaggle/input/datasets/omerfaisal443/aether-data/data")
sys.path.append(str(BASE_DIR))

from data_pipeline.cms_loader import load_cms_data

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"==============================")
print(f"Device Selected: {str(device).upper()}")
if torch.cuda.is_available():
    print(f"GPU Profile: {torch.cuda.get_device_name(0)}")
print(f"==============================\n")

# ---------------------------------------------------------
# 1. High-Performance GPU-Vectorized Dataset
# ---------------------------------------------------------
class CDE_CMS_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list, id_col: str = "provider_id", date_col: str = "claim_date", target_col: str = "denied", scaler=None):
        print("📊 Formatting raw dataframe structures...")
        self.df = df.copy()
        
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df = self.df.sort_values(by=[id_col, date_col]).reset_index(drop=True)
        
        self.features = feature_cols
        
        # 🚨 FIX TYPE SAFETY: Force all feature columns to numeric
        # Coerces strings/objects to NaN, then fills with 0 before scaling
        for col in self.features:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df[self.features] = self.df[self.features].fillna(0)
        
        # 🚨 FIX DATA LEAKAGE: Only fit the scaler if it's not provided
        if scaler is None:
            self.scaler = RobustScaler()
            self.df[self.features] = self.scaler.fit_transform(self.df[self.features])
        else:
            self.scaler = scaler
            self.df[self.features] = self.scaler.transform(self.df[self.features])
        
        self.providers = self.df[id_col].unique()
        num_providers = len(self.providers)
        max_len = self.df.groupby(id_col).size().max()
        input_dim = len(self.features) + 1 
        
        print(f"🧠 Pre-allocating contiguous tensor block for {num_providers} trajectories...")
        padded_X = torch.zeros((num_providers, max_len, input_dim), dtype=torch.float32)
        labels = torch.zeros(num_providers, dtype=torch.float32)
        
        grouped = self.df.groupby(id_col)
        
        for idx, pid in enumerate(tqdm(self.providers, desc="Vector Allocation")):
            seq = grouped.get_group(pid)
            
            X_mat = seq[self.features].values
            y_val = seq[target_col].max()
            
            t_mat = (seq[date_col] - seq[date_col].min()).dt.days.values.astype(np.float32)
            
            if len(np.unique(t_mat)) != len(t_mat):
                t_mat = t_mat + np.arange(len(t_mat)) * 1e-4
                
            X_with_time = np.concatenate([t_mat.reshape(-1, 1), X_mat], axis=1)
            seq_len = len(X_with_time)
            
            padded_X[idx, :seq_len, :] = torch.tensor(X_with_time, dtype=torch.float32)
            
            # 🚨 CDE MATH: Keep the forward-fill padding. 
            # Repeating the last row (including time) ensures the path derivative is zero 
            # during padded steps, correctly "freezing" the hidden state.
            if seq_len < max_len:
                last_row = padded_X[idx, seq_len - 1, :]
                padded_X[idx, seq_len:, :] = last_row.repeat(max_len - seq_len, 1)
                
            labels[idx] = y_val

        print(f"🚀 Pushing tracking tensor layout to {str(device).upper()} for batch interpolation...")
        padded_X = padded_X.to(device)
        
        print("⚙️ Executing parallelized Hermite Cubic Spline interpolation on GPU cores...")
        self.coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(padded_X)
        self.labels = labels.to(device)
        self.input_dim = input_dim

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.coeffs[idx], self.labels[idx]

# ---------------------------------------------------------
# 2. Neural CDE Architecture
# ---------------------------------------------------------
class CDEFunc(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(CDEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, input_dim * hidden_dim)
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, t, h):
        out = self.net(h)
        return out.view(-1, self.hidden_dim, self.input_dim)

class NeuralCDE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1):
        super(NeuralCDE, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.initial = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU()
        )
        
        self.func = CDEFunc(input_dim, hidden_dim)
        
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, coeffs):
        X = torchcde.CubicSpline(coeffs)
        X0 = X.evaluate(X.interval[0])
        h0 = self.initial(X0)
        
        h_T = torchcde.cdeint(X=X,
                              z0=h0,
                              func=self.func,
                              t=X.interval,
                              method='rk4',          
                              options=dict(step_size=1.0)) 
        
        terminal_state = h_T[:, -1, :]
        logits = self.readout(terminal_state)
        return logits.squeeze(-1)

# ---------------------------------------------------------
# 3. Execution & Training Loop
# ---------------------------------------------------------
def train_model():
    print("🚀 Initializing Ultra-SOTA Parallel CDE Training Sequence...")
    
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    
    csv_path = csv_files[0]
    print(f"Loading through pipeline parser: {csv_path}")
    
    raw_df = load_cms_data(
        real_csv=str(csv_path),
        n_rows=5000,
        seed=42
    )
    
    # 🚨 FIX DATA LEAKAGE: Split the raw dataframe BEFORE creating datasets
    train_df = raw_df.sample(frac=0.8, random_state=42)
    val_df = raw_df.drop(train_df.index)
    
    features = [
        "allowed_amount", "billed_amount", "drg_weight", 
        "length_of_stay", "icd_chapter", "procedure_count", 
        "prior_denial_rate", "payer_score"
    ]
    
    # Create train dataset first (fits the scaler ONLY on training data)
    train_dataset = CDE_CMS_Dataset(train_df, feature_cols=features)
    
    # Create val dataset using the scaler from the train dataset
    val_dataset = CDE_CMS_Dataset(val_df, feature_cols=features, scaler=train_dataset.scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = NeuralCDE(input_dim=train_dataset.input_dim, hidden_dim=16).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    
    epochs = 50
    best_loss = float('inf')
    
    print("\n🏁 Commencing Continuous Variable Field Integration Optimization Loop...\n")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_coeffs, batch_labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            
            preds = model(batch_coeffs)
            loss = criterion(preds, batch_labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        
        # Accumulate predictions across the entire validation epoch for accurate metrics
        all_probs = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_coeffs, batch_labels in val_loader:
                preds = model(batch_coeffs)
                v_loss = criterion(preds, batch_labels)
                val_loss += v_loss.item()
                
                probs = torch.sigmoid(preds)
                predicted = (probs > 0.5).float()
                
                all_probs.append(probs.cpu().numpy())
                all_preds.append(predicted.cpu().numpy())
                all_labels.append(batch_labels.cpu().numpy())
                
        val_loss /= len(val_loader)
        
        # Concatenate all batches
        all_probs = np.concatenate(all_probs)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # 📊 Calculate comprehensive evaluation metrics
        val_acc = accuracy_score(all_labels, all_preds) * 100
        val_precision = precision_score(all_labels, all_preds, zero_division=0)
        val_recall = recall_score(all_labels, all_preds, zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        # Handle edge case where an epoch might randomly contain only one class
        if len(np.unique(all_labels)) > 1:
            val_roc_auc = roc_auc_score(all_labels, all_probs)
            val_pr_auc = average_precision_score(all_labels, all_probs)
        else:
            val_roc_auc = 0.0
            val_pr_auc = 0.0
            
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
        val_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "/kaggle/working/best_cde_weights.pth")
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Acc: {val_acc:.2f}% | Prec: {val_precision:.4f} | Rec: {val_recall:.4f} | "
                  f"F1: {val_f1:.4f} | ROC-AUC: {val_roc_auc:.4f} | PR-AUC: {val_pr_auc:.4f} | Spec: {val_specificity:.4f}")

    print("\n✅ Process Finalized. SOTA Weights successfully archived at /kaggle/working/best_cde_weights.pth")

if __name__ == "__main__":
    train_model()