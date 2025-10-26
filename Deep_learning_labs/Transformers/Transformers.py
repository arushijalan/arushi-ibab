import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from Bio import SeqIO
import pandas as pd
import numpy as np
from collections import defaultdict

# Global Configuration
AA_DICT = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
    'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
    'X': 21, '<PAD>': 0
}

MAX_LENGTH = 512 
# Data Loading and Encoding

def parse_fasta(filepath):
    seq_dict = {}
    for record in SeqIO.parse(filepath, "fasta"):
        protein_id = record.id.split('|')[1] if '|' in record.id else record.id
        seq_dict[protein_id] = str(record.seq)
    return seq_dict


def parse_annotations(filepath):
    ann_dict = defaultdict(set)
    df = pd.read_csv(filepath, sep='\t', header=None, names=['protein_id', 'go_term', 'ontology'])
    for _, row in df.iterrows():
        ann_dict[row['protein_id']].add(row['go_term'])
    return ann_dict


def create_go_mapping(annotation_dict):
    all_terms = sorted({term for terms in annotation_dict.values() for term in terms})
    return {term: i for i, term in enumerate(all_terms)}


def sequence_to_int(seq, max_len=MAX_LENGTH):
    seq = seq[:max_len]
    encoded = [AA_DICT.get(aa, AA_DICT['X']) for aa in seq]
    return encoded + [AA_DICT['<PAD>']] * (max_len - len(encoded))


def labels_to_vector(protein_id, ann_dict, go_index):
    label = np.zeros(len(go_index), dtype=np.float32)
    if protein_id in ann_dict:
        for go in ann_dict[protein_id]:
            if go in go_index:
                label[go_index[go]] = 1.0
    return label

# Positional Encoding
def positional_encoding(dim, max_len=MAX_LENGTH):
    pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
    pe = torch.zeros(max_len, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)

# Transformer Model (Functional)
def build_transformer(vocab_size, dim, heads, layers, ff_dim, num_classes, dropout=0.3):
    model = nn.ModuleDict({
        'embedding': nn.Embedding(vocab_size, dim, padding_idx=0),
        'positional': nn.Parameter(positional_encoding(dim), requires_grad=False),
        'encoder': nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation='relu',
                batch_first=True
            ),
            num_layers=layers
        ),
        'fc1': nn.Linear(dim, dim // 2),
        'fc2': nn.Linear(dim // 2, num_classes),
        'relu': nn.ReLU(),
        'dropout': nn.Dropout(dropout)
    })
    return model


def forward_pass(model, src):
    pad_mask = (src == 0)
    embed = model['embedding'](src) * np.sqrt(model['embedding'].embedding_dim)
    embed = embed + model['positional'][:, :embed.size(1), :]
    encoded = model['encoder'](embed, src_key_padding_mask=pad_mask)

    mask = (~pad_mask).unsqueeze(-1).float()
    pooled = (encoded * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-10)

    out = model['dropout'](pooled)
    out = model['relu'](model['fc1'](out))
    out = model['dropout'](out)
    out = torch.sigmoid(model['fc2'](out))
    return out

# Training and Evaluation
def batchify_data(seqs, anns, go_index, batch_size):
    ids = list(seqs.keys())
    np.random.shuffle(ids)
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        seq_batch = [sequence_to_int(seqs[pid]) for pid in batch_ids]
        label_batch = [labels_to_vector(pid, anns, go_index) for pid in batch_ids]
        yield torch.LongTensor(seq_batch), torch.FloatTensor(label_batch)


def train_one_epoch(model, seqs, anns, go_index, optimizer, loss_fn, device, batch_size):
    model.train()
    total_loss = 0
    for seq_batch, label_batch in batchify_data(seqs, anns, go_index, batch_size):
        seq_batch, label_batch = seq_batch.to(device), label_batch.to(device)
        optimizer.zero_grad()
        output = forward_pass(model, seq_batch)
        loss = loss_fn(output, label_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss


def evaluate_model(model, seqs, anns, go_index, loss_fn, device, batch_size):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for seq_batch, label_batch in batchify_data(seqs, anns, go_index, batch_size):
            seq_batch, label_batch = seq_batch.to(device), label_batch.to(device)
            output = forward_pass(model, seq_batch)
            loss = loss_fn(output, label_batch)
            total_loss += loss.item()
    return total_loss


# Inference
def predict_terms(model, seq, go_index, device, threshold=0.01):
    model.eval()
    encoded = sequence_to_int(seq)
    inp = torch.LongTensor([encoded]).to(device)
    with torch.no_grad():
        output = forward_pass(model, inp)
        probs = output.cpu().numpy()[0]
    inv_idx = {v: k for k, v in go_index.items()}
    return [(inv_idx[i], p) for i, p in enumerate(probs) if p >= threshold]


def test_inference(model, fasta_file, go_index, out_file, device):
    seqs = parse_fasta(fasta_file)
    with open(out_file, 'w') as f:
        for pid, seq in seqs.items():
            preds = predict_terms(model, seq, go_index, device)
            preds = sorted(preds, key=lambda x: x[1], reverse=True)[:1500]
            for term, prob in preds:
                f.write(f"{pid}\t{term}\t{prob:.3f}\n")
    print(f"Predictions saved to {out_file}")

# Main Execution
def main():
    train_fasta = "./data/cafa-6-protein-function-prediction/Train/train_sequences.fasta"
    train_terms = "./data/cafa-6-protein-function-prediction/Train/train_terms.tsv"
    test_fasta = ".data/cafa-6-protein-function-prediction/Test/testsuperset.fasta"
    output_path = "./data/cafa-6-protein-function-prediction/output_bilstm.csv"

    print("Loading data...")
    seqs = parse_fasta(train_fasta)
    anns = parse_annotations(train_terms)
    go_index = create_go_mapping(anns)
    print(f"Loaded {len(seqs)} sequences and {len(go_index)} GO terms.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    model = build_transformer(
        vocab_size=len(AA_DICT),
        dim=256,
        heads=8,
        layers=4,
        ff_dim=512,
        num_classes=len(go_index),
        dropout=0.3
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    for epoch in range(20):
        train_loss = train_one_epoch(model, seqs, anns, go_index, optimizer, loss_fn, device, 16)
        val_loss = evaluate_model(model, seqs, anns, go_index, loss_fn, device, 16)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}: Train {train_loss:.4f}, Val {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_transformer.pth')

    print("Training completed.")
    model.load_state_dict(torch.load('best_transformer.pth', map_location=device))
    test_inference(model, test_fasta, go_index, output_path, device)


if __name__ == '__main__':
    main()
