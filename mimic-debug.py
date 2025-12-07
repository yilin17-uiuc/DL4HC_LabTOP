import os
import pickle
import numpy as np
import pandas as pd

DATA_DIR = "eicu_labtop_processed"

def load_split(name):
    path = os.path.join(DATA_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        print(f"{name}.pkl not found at {path}")
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"{name}.pkl: {len(data)} sequences")
    return data

def check_sequences(name, seqs, max_len=1024):
    if seqs is None:
        return
    lengths = [len(x["input_ids"]) for x in seqs]
    print(
        f"{name}: min len={min(lengths)}, max len={max(lengths)}, "
        f"mean len={np.mean(lengths):.1f}"
    )

    # type_ids sanity: same length as input_ids
    bad = sum(len(x["input_ids"]) != len(x["type_ids"]) for x in seqs)
    print(f"{name}: sequences with mismatched type_ids length: {bad}")

    # quick check that some tokens have type_id=1 (labs present)
    num_with_lab_tokens = sum(any(t == 1 for t in x["type_ids"]) for x in seqs)
    print(f"{name}: sequences with any type_id==1 (lab value tokens): {num_with_lab_tokens}")

def check_eval_file(name):
    path = os.path.join(DATA_DIR, f"{name}_eval.pkl")
    if not os.path.exists(path):
        print(f"{name}_eval.pkl not found.")
        return
    with open(path, "rb") as f:
        items = pickle.load(f)
    print(f"{name}_eval.pkl: {len(items)} evaluation items")

    # basic checks on first few
    for i, it in enumerate(items[:5]):
        print(
            f"  Eval[{i}] stay_id={it.get('stay_id')}, "
            f"prompt_len={len(it['prompt_ids'])}, "
            f"label_len={len(it['label_ids'])}, "
            f"event_type={it['event_type']}"
        )

def main():
    print("=== Checking sequence pickles ===")
    train_seqs = load_split("train")
    val_seqs = load_split("val")
    test_seqs = load_split("test")

    check_sequences("train", train_seqs)
    check_sequences("val", val_seqs)
    check_sequences("test", test_seqs)

    print("\n=== Checking eval pickles ===")
    check_eval_file("val")
    check_eval_file("test")

if __name__ == "__main__":
    main()
