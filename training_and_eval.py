import os
import time
import pickle
import random
import numpy as np
from math import isfinite

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

# ---------------- CONFIG ----------------
DATA_DIR = "/content/drive/MyDrive/stay_limit_300_gpt_new_processed_data"  # <-- adjust
MAX_LEN = 1024            # must be <= preprocessor max_len
BATCH_SIZE = 8
LR = 5e-4               # a bit higher for small model
NUM_EPOCHS = 10
PATIENCE = 3
GRAD_CLIP = 1.0


# ---------------- UTILS ----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------- DATASET ----------------
class LabTOPDataset(Dataset):
    """
    Uses only input_ids/type_ids from your preprocessed pkl.
    Training now ignores type_ids for loss masking (LabTOP-style).
    """
    def __init__(self, data_path):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "type_ids": torch.tensor(item["type_ids"], dtype=torch.long),  # kept for potential analysis
        }


# ---------------- COLLATE FN ----------------
def collate_fn(batch, pad_token_id):
    """
    - Truncates to MAX_LEN (from the *right* by default)
    - Pads to the longest length in batch
    - Labels = input_ids shifted internally by GPT2; here we just
      pass input_ids as labels with pad positions set to -100.
    """
    input_ids = [b["input_ids"] for b in batch]

    # truncate sequences (keep last MAX_LEN tokens)
    truncated = []
    for seq in input_ids:
        if len(seq) > MAX_LEN:
            truncated.append(seq[-MAX_LEN:])
        else:
            truncated.append(seq)

    input_ids_pad = torch.nn.utils.rnn.pad_sequence(
        truncated, batch_first=True, padding_value=pad_token_id
    )
    attention_mask = (input_ids_pad != pad_token_id).long()

    labels = input_ids_pad.clone()
    labels[input_ids_pad == pad_token_id] = -100   # ignore pads only

    return input_ids_pad, attention_mask, labels


# ---------------- MODEL (small GPT2-style LM) ----------------
class LabTOPGPT2Small(nn.Module):
    """
    Small GPT-2 style LM:
    - fewer layers / smaller hidden size for your budget
    - still uses GPT2LMHeadModel for correct autoregressive behavior
    """
    def __init__(self, tokenizer, d_model=256, n_heads=4, num_layers=4, max_len=MAX_LEN, dropout=0.1):
        super().__init__()
        vocab_size = len(tokenizer)
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=d_model,
            n_head=n_heads,
            n_layer=num_layers,
            n_positions=max_len,
            n_ctx=max_len,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        self.model = GPT2LMHeadModel(config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        GPT2LMHeadModel will:
        - apply causal masking internally
        - compute next-token cross-entropy loss if labels is provided
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs  # has .logits and .loss

    def generate_next_tokens(self, input_ids, attention_mask=None, max_new_tokens=6, bad_ids=None, eos_id=None):
        """
        Simple greedy generation of a few tokens.
        """
        self.eval()
        device = input_ids.device
        generated = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if input_ids.size(1) > MAX_LEN:
                    input_ids = input_ids[:, -MAX_LEN:]
                    if attention_mask is not None:
                        attention_mask = attention_mask[:, -MAX_LEN:]

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]  # (B, vocab)
                if bad_ids:
                    logits[:, bad_ids] = -1e9

                next_token = torch.argmax(logits, dim=-1)  # (B,)
                next_id = next_token.item()
                generated.append(next_id)

                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                if attention_mask is not None:
                    next_mask_token = torch.ones_like(next_token).unsqueeze(0)
                    attention_mask = torch.cat([attention_mask, next_mask_token], dim=1)

                if eos_id is not None and next_id == eos_id:
                    break

        return generated


# ---------------- DECODE NUMERIC VALUE ----------------
def decode_value(token_ids, tokenizer):
    """
    Char-level decode: keep digits, '.', '-' and parse as float.
    Returns None if parsing fails.
    """
    text = tokenizer.decode(token_ids)
    text = text.replace(" ", "")
    filtered = "".join(ch for ch in text if ch.isdigit() or ch in ".-")
    if filtered == "":
        return None
    try:
        return float(filtered)
    except Exception:
        return None


# ---------------- TRAIN ----------------
def train_model():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(DATA_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    train_dataset = LabTOPDataset(os.path.join(DATA_DIR, "train.pkl"))
    val_dataset = LabTOPDataset(os.path.join(DATA_DIR, "val.pkl"))

    def collate_func(batch):
        return collate_fn(batch, pad_token_id=pad_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_func,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_func,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = LabTOPGPT2Small(tokenizer).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        # ---------- TRAIN ----------
        model.train()
        total_loss = 0.0
        count = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [train]"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            count += 1

        avg_train_loss = total_loss / max(count, 1)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # ---------- VALIDATION ----------
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [val]"):
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                val_loss += loss.item()
                val_count += 1

        avg_val_loss = val_loss / max(val_count, 1)
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")

        # ---------- EARLY STOP ----------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            ckpt_path = os.path.join(DATA_DIR, "labtop_small_gpt2.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model to {ckpt_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print("Training complete.")


# ---------------- EVALUATION ----------------
def evaluate_model():
    test_eval_path = os.path.join(DATA_DIR, "test_eval.pkl")
    ckpt_path = os.path.join(DATA_DIR, "labtop_small_gpt2.pth")

    if not os.path.exists(test_eval_path):
        print("test_eval.pkl not found; nothing to evaluate.")
        return
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} not found; train first.")
        return

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(DATA_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(test_eval_path, "rb") as f:
        test_data = pickle.load(f)

    max_eval_examples = 5000
    if len(test_data) > max_eval_examples:
        test_data = test_data[:max_eval_examples]
    print(f"Running inference on {len(test_data)} examples...")

    model = LabTOPGPT2Small(tokenizer).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    use_amp = (device.type == "cuda")

    # tokens we never want as value predictions
    bad_tokens = ["labevent", "inputevent", "outputevent", "gender", "age", "race"]
    vocab = tokenizer.get_vocab()
    bad_ids = [tokenizer.convert_tokens_to_ids(t) for t in bad_tokens if t in vocab]

    eoe_id = tokenizer.convert_tokens_to_ids("[EOE]")
    max_new_tokens = 6

    predictions = []
    ground_truths = []
    itemids = []
    event_types = []
    example_indices = []

    start_time = time.time()

    for idx, item in enumerate(tqdm(test_data, desc="Evaluating")):
        prompt_ids = item["prompt_ids"]
        true_val = item["valuenum"]
        itemid = item.get("itemid", None)
        e_type = item.get("event_type", "unknown")

        # truncate prompt from the left to MAX_LEN
        if len(prompt_ids) > MAX_LEN:
            prompt_ids = prompt_ids[-MAX_LEN:]

        input_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(device)
        attn_mask = torch.ones_like(input_tensor, dtype=torch.long).to(device)

        if idx < 3:
            decoded_prompt_tail = tokenizer.decode(prompt_ids[-80:])
            print(f"\n=== Debug sample {idx} ===")
            print(f"Prompt tail: {decoded_prompt_tail}")
            print(f"True value: {true_val}, itemid: {itemid}, event_type: {e_type}")

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                generated_ids = model.generate_next_tokens(
                    input_ids=input_tensor,
                    attention_mask=attn_mask,
                    max_new_tokens=max_new_tokens,
                    bad_ids=bad_ids,
                    eos_id=eoe_id,
                )

        pred_val = decode_value(generated_ids, tokenizer)
        if pred_val is None or not isfinite(pred_val) or abs(pred_val) > 1e4:
            continue

        if idx < 3:
            decoded_generated = tokenizer.decode(generated_ids)
            print(f"Generated tokens: {decoded_generated}")
            print(f"Decoded pred_val: {pred_val}")

        predictions.append(pred_val)
        ground_truths.append(true_val)
        itemids.append(itemid)
        event_types.append(e_type)
        example_indices.append(idx)

    if len(predictions) == 0:
        print("No valid predictions generated.")
        return

    preds = np.array(predictions)
    truths = np.array(ground_truths)

    mae = np.mean(np.abs(preds - truths))
    rmse = np.sqrt(np.mean((preds - truths) ** 2))

    p1 = np.percentile(truths, 1)
    p99 = np.percentile(truths, 99)
    range_val = p99 - p1 if p99 > p1 else 1e-6
    nmae = mae / range_val

    denom = (np.abs(truths) + np.abs(preds))
    mask = denom > 0
    if np.sum(mask) > 0:
        smape = np.mean(2 * np.abs(preds[mask] - truths[mask]) / denom[mask])
    else:
        smape = 0.0

    elapsed = time.time() - start_time
    print(f"\nEvaluated {len(preds)} samples in {elapsed:.2f} seconds.")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test NMAE: {nmae:.4f}")
    print(f"Test SMAPE: {smape:.4f}")

    # optional: save sample predictions
    csv_path = os.path.join(DATA_DIR, "test_predictions_sample_small_gpt2.csv")
    print(f"Saving predictions to: {csv_path}")
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "itemid", "event_type", "true_value", "pred_value"])
        for i, iid, et, truth, pred in zip(example_indices, itemids, event_types, truths, preds):
            writer.writerow([i, iid, et, truth, pred])


if __name__ == "__main__":
    # First run training, then run evaluation
    ##train_model()
    evaluate_model()
