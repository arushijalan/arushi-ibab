import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
import pandas as pd
import numpy as np
from collections import defaultdict

# Amino acid token mapping
AMINO_ACID_MAP = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
    'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
    'X': 21, '<PAD>': 0
}

MAX_LENGTH = 512


def read_fasta_sequences(fasta_path):
    protein_dict = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        protein_id = record.id.split('|')[1] if '|' in record.id else record.id
        protein_dict[protein_id] = str(record.seq)
    return protein_dict

#Load protein-GO term mappings
def read_go_annotations(annotation_path):
    annotation_dict = defaultdict(set)
    df = pd.read_csv(annotation_path, sep='\t', header=None, names=['protein', 'term', 'ontology'])
    for _, row in df.iterrows():
        annotation_dict[row['protein']].add(row['term'])
    return annotation_dict

# Create mapping from GO term index
def create_go_index(annotation_dict):
    
    all_terms = sorted({term for terms in annotation_dict.values() for term in terms})
    return {term: idx for idx, term in enumerate(all_terms)}

# Convert amino acid sequence to numerical tensor
def encode_protein_sequence(seq, max_len=MAX_LENGTH):
    
    seq = seq[:max_len]
    encoded_seq = [AMINO_ACID_MAP.get(aa, AMINO_ACID_MAP['X']) for aa in seq]
    padded_seq = encoded_seq + [AMINO_ACID_MAP['<PAD>']] * (max_len - len(encoded_seq))
    return padded_seq

# Create binary vector for GO term annotations
def encode_go_labels(protein_id, annotation_dict, term_index):
    
    label_vec = np.zeros(len(term_index), dtype=np.float32)
    if protein_id in annotation_dict:
        for term in annotation_dict[protein_id]:
            if term in term_index:
                label_vec[term_index[term]] = 1.0
    return label_vec

# Convert data into tensors for model input
def prepare_dataset(sequence_dict, annotation_dict, term_index):
   
    protein_ids = list(sequence_dict.keys())
    encoded_sequences, label_vectors = [], []

    for pid in protein_ids:
        encoded_sequences.append(encode_protein_sequence(sequence_dict[pid]))
        label_vectors.append(encode_go_labels(pid, annotation_dict, term_index))

    encoded_sequences = torch.LongTensor(encoded_sequences)
    label_vectors = torch.FloatTensor(label_vectors)
    return protein_ids, encoded_sequences, label_vectors

#Build BiLSTM model using sequential API
def create_bilstm_model(vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, dropout_rate=0.3):
   
    embedding_layer = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    lstm_layer = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                         batch_first=True, bidirectional=True,
                         dropout=dropout_rate if num_layers > 1 else 0)
    dropout = nn.Dropout(dropout_rate)
    fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
    fc2 = nn.Linear(hidden_dim, num_classes)
    activation = nn.ReLU()
    sigmoid = nn.Sigmoid()

    def model_forward(x):
        embedded = dropout(embedding_layer(x))
        lstm_out, _ = lstm_layer(embedded)
        pooled, _ = torch.max(lstm_out, dim=1)
        output = dropout(pooled)
        output = activation(fc1(output))
        output = dropout(output)
        output = sigmoid(fc2(output))
        return output

    return model_forward, [embedding_layer, lstm_layer, fc1, fc2, dropout]

# Run one training epoch
def train_one_epoch(forward_fn, params, dataloader, criterion, optimizer, device):
 
    for layer in params:
        layer.train()
    epoch_loss = 0.0

    for seqs, labels in dataloader:
        seqs, labels = seqs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = forward_fn(seqs)
        loss = criterion(preds, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(sum([list(p.parameters()) for p in params], []), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

# Validate model on held-out data
def evaluate_model(forward_fn, params, dataloader, criterion, device):
    
    for layer in params:
        layer.eval()
    val_loss = 0.0

    with torch.no_grad():
        for seqs, labels in dataloader:
            seqs, labels = seqs.to(device), labels.to(device)
            preds = forward_fn(seqs)
            loss = criterion(preds, labels)
            val_loss += loss.item()

    return val_loss / len(dataloader)

# Predict GO terms for a single protein
def predict_terms(forward_fn, params, seq, term_index, device, threshold=0.01):
    
    for layer in params:
        layer.eval()

    encoded = torch.LongTensor([encode_protein_sequence(seq)]).to(device)
    with torch.no_grad():
        probs = forward_fn(encoded).cpu().numpy()[0]

    idx_to_term = {v: k for k, v in term_index.items()}
    predictions = [(idx_to_term[i], p) for i, p in enumerate(probs) if p >= threshold]
    return sorted(predictions, key=lambda x: x[1], reverse=True)

# Evaluate model on test sequences and save predictions
def evaluate_on_test_set(forward_fn, params, test_fasta, term_index, output_file, device, threshold=0.01, top_n=1500):
    
    test_sequences = read_fasta_sequences(test_fasta)
    with open(output_file, 'w') as f_out:
        for pid, seq in test_sequences.items():
            preds = predict_terms(forward_fn, params, seq, term_index, device, threshold)
            for go_term, prob in preds[:top_n]:
                f_out.write(f"{pid}\t{go_term}\t{prob:.3f}\n")
    print(f"Predictions saved to {output_file}")


def main():
    train_fasta = "./data/cafa-6-protein-function-prediction/Train/train_sequences.fasta"
    train_terms = "./data/cafa-6-protein-function-prediction/Train/train_terms.tsv"
    test_fasta = ".data/cafa-6-protein-function-prediction/Test/testsuperset.fasta"
    output_path = "./data/cafa-6-protein-function-prediction/output_bilstm.csv"

    seq_dict = read_fasta_sequences(train_fasta)
    annotation_dict = read_go_annotations(train_terms)
    term_index = create_go_index(annotation_dict)
    print(f"Loaded {len(seq_dict)} sequences with {len(term_index)} GO terms.")

    protein_ids, encoded_seqs, label_vecs = prepare_dataset(seq_dict, annotation_dict, term_index)
    dataset = list(zip(encoded_seqs, label_vecs))

    split_idx = int(0.8 * len(dataset))
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    forward_fn, model_layers = create_bilstm_model(
        vocab_size=len(AMINO_ACID_MAP),
        embed_dim=128,
        hidden_dim=256,
        num_classes=len(term_index),
        num_layers=2,
        dropout_rate=0.3
    )

    all_params = [p for layer in model_layers for p in layer.parameters()]
    criterion = nn.BCELoss()
    optimizer = optim.Adam(all_params, lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    num_epochs = 20

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(forward_fn, model_layers, train_loader, criterion, optimizer, device)
        val_loss = evaluate_model(forward_fn, model_layers, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save([layer.state_dict() for layer in model_layers], "bilstm_model_weights.pth")
            print(f"Saved model at epoch {epoch+1}")

    print("Training complete. Loading best model...")
    saved_states = torch.load("bilstm_model_weights.pth", map_location=device)
    for layer, state in zip(model_layers, saved_states):
        layer.load_state_dict(state)

    evaluate_on_test_set(forward_fn, model_layers, test_fasta, term_index, output_path, device)


if __name__ == "__main__":
    main()
