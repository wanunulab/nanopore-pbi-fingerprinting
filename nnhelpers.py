import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from typing import List

class ProteinEventDataset(Dataset):
    """
    Dataset for nanopore protein events.
    Each event: variable-length sequence of steps, each step has (mean, std, duration).
    Target: list of chunk indices (e.g. [1,2,1,2,1,2] for R1).
    """
    def __init__(self, events, targets, target_types, row_ids):
        self.events           = [torch.from_numpy(e).float() for e in events]
        self.targets          = [torch.tensor(t, dtype=torch.long) for t in targets]
        self.target_prot_type = [torch.tensor(t, dtype=torch.long) for t in target_types]
        self.row_ids          = list(row_ids)

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        # return row_id so collate and inference can see it
        return (
            self.events[idx],
            self.targets[idx],
            self.target_prot_type[idx],
            self.row_ids[idx]
        )

def collate_fn(batch):
    """
    Pads events and targets for batching.
    Returns padded_x (B, T_max, 3), x_lengths, padded_targets (sum of target lengths), target_lengths.
    Notes: For CTC, targets should be concatenated in a flat tensor.
    """
    # Separate sequences and targets
    xs, ys, ts, ids = zip(*batch)
    x_lengths = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    y_lengths = torch.tensor([y.size(0) for y in ys], dtype=torch.long)

    padded_x = pad_sequence(xs, batch_first=True, padding_value=0.0)
    flat_y   = torch.cat(ys)
    prot_ts  = torch.tensor(ts, dtype=torch.long)
    row_ids  = torch.tensor(ids, dtype=torch.long)

    return padded_x, flat_y, x_lengths, y_lengths, prot_ts, row_ids

# =============================
# 2. Model Definition
# =============================


class BiLSTMClassifier(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 num_classes: int,
                 dropout: float = 0.25):
        super().__init__()
        # 1) The BiLSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers>1 else 0.0
        )
        # 2) A small FC "stack" on top of the final-layer hidden states
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc2     = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        # Pack / run through LSTM
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(),
                                                          batch_first=True,
                                                          enforce_sorted=False)
        
        packed_out, (h_n, _) = self.lstm(packed)
        # h_n is (num_layers*2, B, hidden_dim)
        # grab last-layer forward + backward
        # forward = h_n[-2], backward = h_n[-1]
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h     = torch.cat([h_fwd, h_bwd], dim=1)  # (B, 2*hidden_dim)

        # FC stack → logits over protein types
        h = self.dropout(h)
        h = F.relu(self.fc1(h))
        logits = self.fc2(h)                      # (B, num_classes)
        return logits

# =============================
# 3. Training Loop
# =============================


def train(model, dataloader, optimizer, device):
    model.train()
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    total_loss = 0.0

    for x_batch, y_flat, x_lengths, y_lengths,_ in dataloader:
        x_batch = x_batch.to(device)
        y_flat = y_flat.to(device)
        x_lengths = x_lengths.to(device)
        y_lengths = y_lengths.to(device)

        optimizer.zero_grad()
        # Forward
        log_probs = model(x_batch, x_lengths)  # (B, T, C)
        # CTC expects (T, B, C)
        log_probs = log_probs.permute(1, 0, 2)

        # Compute CTC loss
        loss = ctc_loss(log_probs, y_flat, x_lengths, y_lengths)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
def train_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, total_examples = 0.0, 0

    for x, _seq_targets, x_lens, _y_lens, prot_ts, _row_ids in loader:
        x, x_lens, labels = x.to(device), x_lens.to(device), prot_ts.to(device)

        optimizer.zero_grad()
        logits = model(x, x_lens)                # (B, num_classes)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss     += loss.item() * labels.size(0)
        total_examples += labels.size(0)

    return total_loss / total_examples

# =============================
# 4. Inference 
# =============================

def eval_classifier(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, _seq, x_lens, _y_lens , prot_ts, _ in loader:
            x, x_lens, labels = x.to(device), x_lens.to(device), prot_ts.to(device)
            logits = model(x, x_lens)
            preds  = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct/total

def eval_cls_loss(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for x, _seq, x_lens, _y_lens, labels, _ids in loader:
            x, x_lens, labels = x.to(device), x_lens.to(device), labels.to(device)
            logits = model(x, x_lens)                # (B, num_classes)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total      += labels.size(0)
    return total_loss / total

def shorten_signal_by_deviation(signal, deviation_threshold,depth=0):
    
    if depth>500:
        return signal
    diff=np.abs(np.diff(signal,prepend=[np.inf],axis=0))
    
    if signal.shape[0]<=1:
        return signal
    if np.min(diff)>deviation_threshold:
        return signal
    min_idx=np.argmin(diff)
    
    return shorten_signal_by_deviation(np.concatenate([signal[:min_idx-1],[np.mean(signal[min_idx-1:min_idx+1],axis=0)],signal[min_idx+1:]],axis=0),deviation_threshold=deviation_threshold,depth=depth+1)

def eventwise_standardise(means: List[np.ndarray],
                          stds:  List[np.ndarray],
                          dwells: List[np.ndarray],
                          baselines: List[float]) -> List[np.ndarray]:
    """
    Returns Xn - a list of 2-D arrays, one per event, where column 0
    (the step mean) has been renormalized per-event.  Shape (T_i, 1).
    
    - No global StandardScaler is used.
    """
    Xn = []
    for m, s, d, b in zip(means, stds, dwells, baselines):
        
        event = np.vstack((20 * m / b - 2, 1)).T   # shape (T, 2). 
        # m/b is in 0.05-0.15 range for this particular experimental setup. 
        # 20 * m/b -2 renormalizes the range to roughly -1 to 1 without absolute capping.

        step_means = event[:, 0]                       # column we care about
        mu   = step_means.mean()
        
        norm_means = step_means - mu        # normalizes event steps have a zero average 
        Xn.append(norm_means[:, None]) 
    return Xn


def load_protein_data(paths, label_map_int, target_type_map_int,
                      sampling_seed, count_per_class, val_frac=0.2,drop_files=[],scaler=None):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    df = pd.concat([pd.read_pickle(p) for p in paths], ignore_index=True)
    df = df[(df["duration"]>0.001)&(df["duration"]<0.1)&(df["fine_step_count"]>20)].copy()
    df["label_map_int"]       = df["sample_type_paper"].map(label_map_int)
    df["target_type_map_int"] = df["sample_type_paper"].map(target_type_map_int)
    
    if drop_files:
        df=df.loc[~df["filename"].apply(lambda f: True in [filename in f for filename in drop_files])]
    
    
    for gp,_df in df.groupby("sample_type_paper"):
        print(f"{gp}\t", len(_df))
    samp = df.groupby("sample_type_paper") \
             .sample(count_per_class, random_state=sampling_seed)

    train_df, val_df = train_test_split(
        samp, test_size=val_frac,
        stratify=samp["sample_type_paper"],
        random_state=sampling_seed
    )
    
    
    
    scaler = StandardScaler()
    def df_to_xy(df_,scaler,fit=True):
        means  = df_["fine_means"]
        stds   = df_["fine_stds"]
        dwells = df_["fine_dwells"]
        baselines=df_["baseline"]
        # means=[shorten_signal_by_deviation(mean_steps, deviation_threshold=5) for mean_steps in means.values]
        Xn = eventwise_standardise(means, stds, dwells, baselines)
        # Xncat=np.concatenate(Xn,axis=0)
        # print(np.mean(Xncat),np.std(Xncat))
        Ys       = df_["label_map_int"].tolist()
        Ts       = df_["target_type_map_int"].tolist()
        row_ids  = df_.index.to_list()             # <-- original DataFrame row indices
        return Xn, Ys, Ts, row_ids

    return (
        *df_to_xy(train_df,scaler),train_df,
        *df_to_xy(val_df,scaler,fit=False),val_df,scaler
    )
def load_unlabeled_data(paths,scaler,drop_files=[],filter_functions=[]):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    df = pd.concat([pd.read_pickle(p) for p in paths], ignore_index=True)
    df = df[(df["duration"]>0.001)&(df["duration"]<0.1)&(df["fine_step_count"]>20)].copy()
    df["label_map_int"]=df["duration"].apply(lambda x:[-1,-1,-1])
    df["target_type_map_int"]=-1

    if drop_files:
        df=df.loc[~df["filename"].apply(lambda f: True in [filename in f for filename in drop_files])]
    if filter_functions:
        for filter_function in filter_functions:
            df=df.loc[df.apply(filter_function)]
            
    for gp,_df in df.groupby("sample_type_paper"):
        print(f"{gp}\t", len(_df))  
    def df_to_xy(df_,scaler,fit=False):
        means  = df_["fine_means"]
        stds   = df_["fine_stds"]
        dwells = df_["fine_dwells"]
        baselines=df_["baseline"]
        
        Xn = eventwise_standardise(means, stds, dwells, baselines)
        Ys       = df_["label_map_int"].tolist()
        Ts       = df_["target_type_map_int"].tolist()
        row_ids  = df_.index.to_list()             # <-- original DataFrame row indices
        return Xn, Ys, Ts, row_ids
    return (*df_to_xy(df,scaler,fit=True),df,scaler)
    
    
    
    import math

# 10³ exponents ➜ prefixes (ISO 80000-1)
_SI_PREFIXES = {
     24: 'Y', 21: 'Z', 18: 'E', 15: 'P', 12: 'T',  9: 'G',
      6: 'M',  3: 'k',  0: ' ',  -3: 'm', -6: 'µ', -9: 'n',
    -12: 'p', -15: 'f', -18: 'a', -21: 'z', -24: 'y',
}

def si_format(value: float, precision: int = 3,unit='') -> str:
    """
    Return `value` as a string with an SI metric prefix.
    
    >>> si_format(123456)
    '123 k'
    >>> si_format(0.0000042, 2)
    '4.2 µ'
    """
    # if value is pd.NaN:
    #     value=0
    # Special-case zero (log10 is undefined there)
    if value == 0:
        return f"{0:.{precision}g} {unit}".rstrip()       # -> '0'
    
    # Work in absolute terms, but remember original sign
    sign = '-' if value < 0 else ''
    value = abs(value)

    # Pick the nearest power of 10 that is a multiple of 3
    exponent = int(math.floor(math.log10(value) / 3) * 3)
    exponent = max(min(exponent, 24), -24)           # clamp to table

    scaled = value / (10 ** exponent)
    prefix = _SI_PREFIXES[exponent]

    # `g` drops trailing zeros, keeps scientific for small precisions
    return f"{sign}{scaled:.{precision}g} {prefix}{unit}".rstrip()