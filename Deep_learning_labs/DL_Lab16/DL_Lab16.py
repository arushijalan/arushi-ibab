# Implement sequence to sequence learning using LSTM and GRU. 
# You can use the the English to Hindi & Hindi to English corpus dataset 
# https://www.kaggle.com/code/aiswaryaramachandran/english-to-hindi-neural-machine-translation 
# Study the performance between LSTM and GRU.
# Demonstrate how both LSTM and GRU solves the vanishing gradient problem.

import os
import random
import re
import string
from string import digits

import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset

# Reproducibility
SEED = 999
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Dataset
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, input_token_index, target_token_index,
                 max_len_src, max_len_tgt):
        self.src = src_sentences.tolist()
        self.tgt = tgt_sentences.tolist()
        self.input_token_index = input_token_index
        self.target_token_index = target_token_index
        self.max_len_src = max_len_src
        self.max_len_tgt = max_len_tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        # convert to indices, pad with 0
        src = self.src[idx].split()
        tgt = self.tgt[idx].split()

        enc = np.zeros(self.max_len_src, dtype=np.int64)
        for i, w in enumerate(src[: self.max_len_src]):
            enc[i] = self.input_token_index.get(w, 0)

        dec_in = np.zeros(self.max_len_tgt, dtype=np.int64)
        dec_target = np.zeros(self.max_len_tgt, dtype=np.int64)

        for t, w in enumerate(tgt[: self.max_len_tgt]):
            dec_in[t] = self.target_token_index.get(w, 0)
        # target is next token (shifted left by 1)
        for t in range(1, min(len(tgt), self.max_len_tgt)):
            dec_target[t - 1] = self.target_token_index.get(tgt[t], 0)

        return torch.from_numpy(enc), torch.from_numpy(dec_in), torch.from_numpy(dec_target)

# Models
class Encoder(nn.Module):
    def __init__(self, input_vocab_size, emb_size, hidden_size, padding_idx=0, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, emb_size, padding_idx=padding_idx)
        self.bidirectional = bidirectional
        if bidirectional:
            self.lstm = nn.LSTM(emb_size, hidden_size // 2, batch_first=True, bidirectional=True)
        else:
            self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)

    def forward(self, x):
        emb = self.embedding(x)
        outputs, (h, c) = self.lstm(emb)
        return h, c


class Decoder(nn.Module):
    def __init__(self, target_vocab_size, emb_size, hidden_size, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(target_vocab_size, emb_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, target_vocab_size)

    def forward_step(self, x, states):

        emb = self.embedding(x)  # (batch,1,emb)
        outputs, states = self.lstm(emb, states)  # outputs: (batch,1,hidden)
        logits = self.fc(outputs.squeeze(1))  # (batch, vocab)
        return logits, states

    def forward(self, x, states):
        # support full-sequence forward (for validation convenience)
        emb = self.embedding(x)
        outputs, states = self.lstm(emb, states)
        logits = self.fc(outputs)  # (batch, seq_len, vocab)
        return logits, states

# Decoding Utilities
def greedy_decode(encoder, decoder, src_seq, max_len_tgt, target_token_index, reverse_target_index, device):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        src_seq = src_seq.to(device)
        h, c = encoder(src_seq)
        states = (h, c)

        start_idx = target_token_index.get('START_', 0)
        cur_input = torch.LongTensor([[start_idx]]).to(device)
        decoded_words = []
        for _ in range(max_len_tgt):
            logits, states = decoder.forward_step(cur_input, states)  # (batch=1, vocab)
            topi = logits.argmax(dim=-1).item()
            # if topi is padding or unknown -> stop
            if topi == 0:
                break
            word = reverse_target_index.get(topi, '')
            if word == '_END':
                break
            if word == 'START_':
                # skip start token in output sequence
                cur_input = torch.LongTensor([[topi]]).to(device)
                continue
            decoded_words.append(word)
            cur_input = torch.LongTensor([[topi]]).to(device)
    return ' '.join(decoded_words)


def beam_search_decode(encoder, decoder, src_seq, max_len_tgt, target_token_index, reverse_target_index, device, beam_width=3):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        src_seq = src_seq.to(device)
        h, c = encoder(src_seq)
        # states for each beam: we store (score, token_seq, (h,c))
        start_idx = target_token_index.get('START_', 0)
        init_state = (h, c)
        beams = [(0.0, [start_idx], init_state)]  # log-prob score, token list, states

        for _ in range(max_len_tgt):
            new_beams = []
            for score, seq, states in beams:
                last = seq[-1]
                if last == target_token_index.get('_END', -1):
                    # already finished
                    new_beams.append((score, seq, states))
                    continue
                cur_input = torch.LongTensor([[last]]).to(device)
                logits, new_states = decoder.forward_step(cur_input, states)
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)  # (vocab,)
                topk = torch.topk(log_probs, beam_width)
                for k in range(beam_width):
                    tok = topk.indices[k].item()
                    tok_score = topk.values[k].item()
                    new_seq = seq + [tok]
                    new_score = score + tok_score
                    new_beams.append((new_score, new_seq, new_states))
            # keep top beam_width beams
            new_beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]
            beams = new_beams

        # pick best completed beam (or best overall)
        best = beams[0]
        toks = best[1]
        words = []
        for t in toks:
            if t == target_token_index.get('START_', -1):
                continue
            if t == target_token_index.get('_END', -1) or t == 0:
                break
            words.append(reverse_target_index.get(t, ''))
        return ' '.join(words)

# Training / Validation (stepwise decoder)
def train_epoch_stepwise(encoder, decoder, dataloader, enc_optimizer, dec_optimizer, criterion, device, teacher_forcing_ratio=0.5, grad_clip=1.0):
    encoder.train()
    decoder.train()
    total_loss = 0.0
    total_tokens = 0
    for enc_inp, dec_inp, dec_target in dataloader:
        enc_inp = enc_inp.to(device)          # (batch, max_src)
        dec_inp = dec_inp.to(device)          # (batch, max_tgt)  contains START_ at t=0 usually
        dec_target = dec_target.to(device)    # (batch, max_tgt) targets shifted

        batch_size = enc_inp.size(0)
        max_t = dec_inp.size(1)

        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()

        h, c = encoder(enc_inp)
        states = (h, c)

        # start token indices for each batch
        start_idx = dec_inp[:, 0].unsqueeze(1)  # (batch,1)
        input_tok = start_idx  # initial decoder input

        loss = 0.0
        # iterate timesteps
        for t in range(max_t):
            logits, states = decoder.forward_step(input_tok, states)  # logits: (batch, vocab)
            # target at time t is dec_target[:, t]
            target_t = dec_target[:, t]  # (batch,)
            # accumulate loss (CrossEntropy accepts (N, C) and (N,))
            loss_t = criterion(logits, target_t)
            loss = loss + loss_t

            # decide next input for each sample: teacher forcing or use predicted
            use_teacher = (torch.rand(batch_size) < teacher_forcing_ratio).to(device)
            predicted = logits.argmax(dim=-1).unsqueeze(1)  # (batch,1)
            ground = dec_inp[:, t].unsqueeze(1)  # ground truth dec input (note: dec_inp includes START_, tokens)
            # choose next input per sample
            next_input = torch.where(use_teacher.unsqueeze(1), ground, predicted)
            input_tok = next_input.detach()

        # backprop
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
        enc_optimizer.step()
        dec_optimizer.step()

        # account loss weighted by batch
        total_loss += loss.item() * batch_size
        # total tokens approximate
        total_tokens += batch_size * max_t

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def validate_epoch_fullseq(encoder, decoder, dataloader, criterion, device):
    encoder.eval()
    decoder.eval()
    total_loss = 0.0
    with torch.no_grad():
        for enc_inp, dec_inp, dec_target in dataloader:
            enc_inp = enc_inp.to(device)
            dec_inp = dec_inp.to(device)
            dec_target = dec_target.to(device)
            h, c = encoder(enc_inp)
            logits, _ = decoder(dec_inp, (h, c))  # (batch, seq_len, vocab)
            logits_flat = logits.view(-1, logits.size(-1))
            target_flat = dec_target.view(-1)
            loss = criterion(logits_flat, target_flat)
            total_loss += loss.item() * enc_inp.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

# Main
def main():
    csv_path = "/home/ibab/PycharmProjects/3rd_Sem/data/Hindi_English_Truncated_Corpus.csv"
    sample_n = 25000
    random_state = 999

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------- Load & preprocess ----------
    lines = pd.read_csv(csv_path, encoding='utf-8')
    lines = lines[lines['source'] == 'ted']
    lines = lines[~pd.isnull(lines['english_sentence'])]
    lines = lines.drop_duplicates()
    lines = lines.sample(n=sample_n, random_state=random_state)

    # lower, remove apostrophes & punctuation & digits
    lines['english_sentence'] = lines['english_sentence'].str.lower().apply(lambda x: re.sub("'", '', x))
    lines['hindi_sentence'] = lines['hindi_sentence'].str.lower().apply(lambda x: re.sub("'", '', x))

    exclude = set(string.punctuation)
    lines['english_sentence'] = lines['english_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
    lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))

    remove_digits = str.maketrans('', '', digits)
    lines['english_sentence'] = lines['english_sentence'].apply(lambda x: x.translate(remove_digits))
    lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x: x.translate(remove_digits))
    lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x: re.sub("[२३०८१५७९४६]", "", x))
    lines['english_sentence'] = lines['english_sentence'].apply(lambda x: re.sub(" +", " ", x.strip()))
    lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x: re.sub(" +", " ", x.strip()))

    # ensure start/end tokens
    lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x: 'START_ ' + x + ' _END')

    # build vocab sets
    all_eng_words = set()
    for eng in lines['english_sentence']:
        all_eng_words.update(eng.split())

    all_hindi_words = set()
    for hin in lines['hindi_sentence']:
        all_hindi_words.update(hin.split())

    # safety: force START_ and _END
    all_hindi_words.update(['START_', '_END'])

    # filter by length
    lines['length_eng_sentence'] = lines['english_sentence'].apply(lambda x: len(x.split(" ")))
    lines['length_hin_sentence'] = lines['hindi_sentence'].apply(lambda x: len(x.split(" ")))
    lines = lines[lines['length_eng_sentence'] <= 20]
    lines = lines[lines['length_hin_sentence'] <= 20]

    max_length_src = int(lines['length_eng_sentence'].max())
    max_length_tgt = int(lines['length_hin_sentence'].max())
    print("max_length_src:", max_length_src, "max_length_tgt:", max_length_tgt)

    input_words = sorted(list(all_eng_words))
    target_words = sorted(list(all_hindi_words))
    num_encoder_tokens = len(input_words)
    num_decoder_tokens = len(target_words) + 1  # +1 reserved (padding index 0)

    input_token_index = {word: i + 1 for i, word in enumerate(input_words)}
    target_token_index = {word: i + 1 for i, word in enumerate(target_words)}
    reverse_input_index = {i: w for w, i in input_token_index.items()}
    reverse_target_index = {i: w for w, i in target_token_index.items()}

    lines = shuffle(lines, random_state=SEED)
    X, y = lines['english_sentence'], lines['hindi_sentence']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    train_dataset = TranslationDataset(X_train, y_train, input_token_index, target_token_index, max_length_src, max_length_tgt)
    valid_dataset = TranslationDataset(X_test, y_test, input_token_index, target_token_index, max_length_src, max_length_tgt)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Model hyperparams
    latent_dim = 300
    encoder = Encoder(input_vocab_size=num_encoder_tokens + 1, emb_size=latent_dim, hidden_size=latent_dim, padding_idx=0, bidirectional=False).to(device)
    decoder = Decoder(target_vocab_size=num_decoder_tokens, emb_size=latent_dim, hidden_size=latent_dim, padding_idx=0).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)

    # Training settings
    epochs = 30
    teacher_forcing_ratio = 0.5
    best_val = float('inf')

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch_stepwise(encoder, decoder, train_loader, enc_optimizer, dec_optimizer,
                                         criterion, device, teacher_forcing_ratio=teacher_forcing_ratio, grad_clip=1.0)
        val_loss = validate_epoch_fullseq(encoder, decoder, valid_loader, criterion, device)
        print(f"Epoch {epoch}/{epochs}  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'input_token_index': input_token_index,
                'target_token_index': target_token_index,
                'reverse_target_index': reverse_target_index,
                'max_length_src': max_length_src,
                'max_length_tgt': max_length_tgt,
            }, 'nmt_pytorch_stepwise_best.pt')

    # Inference / Sample Outputs

    print("\nSample predictions on random training examples:")
    for i in random.sample(range(len(train_dataset)), 10):
        enc_inp, dec_inp, dec_target = train_dataset[i]
        input_sentence = train_dataset.src[i]
        target_sentence = train_dataset.tgt[i]
        decoded_greedy = greedy_decode(encoder, decoder, enc_inp.unsqueeze(0), max_length_tgt,
                                       target_token_index, reverse_target_index, device)
        decoded_beam3 = beam_search_decode(encoder, decoder, enc_inp.unsqueeze(0), max_length_tgt,
                                           target_token_index, reverse_target_index, device, beam_width=3)
        print(f"Input English sentence: {input_sentence}")
        print(f"Actual Hindi Translation: {target_sentence}")
        print(f"Predicted (greedy): {decoded_greedy}")
        print(f"Predicted (beam=3): {decoded_beam3}")
        print("-" * 60)


if __name__ == "__main__":
    main()
