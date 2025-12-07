import os
import pickle
from collections import Counter
from transformers import AutoTokenizer

OUT_DIR = "stay_limit_300_gpt_new_processed_data"


def load_pkl(name):
    path = os.path.join(OUT_DIR, name)
    with open(path, "rb") as f:
        return pickle.load(f)


def sanity_check():
    # 1. Basic files exist
    for fn in ["train.pkl", "val.pkl", "test.pkl", "val_eval.pkl", "test_eval.pkl"]:
        path = os.path.join(OUT_DIR, fn)
        print(f"{fn}: {'FOUND' if os.path.exists(path) else 'MISSING'}")

    # 2. Load small slices
    train = load_pkl("train.pkl")[:200]
    val = load_pkl("val.pkl")[:200]
    val_eval = load_pkl("val_eval.pkl")[:500]

    print(f"\nTrain seqs (sampled): {len(train)}")
    print(f"Val   seqs (sampled): {len(val)}")
    print(f"Val_eval items (sampled): {len(val_eval)}")

    # 3. Check sequence structure
    for name, data in [("train", train), ("val", val)]:
        bad_len = 0
        max_len = 0
        type_id_mismatch = 0
        stays = set()
        for x in data:
            stays.add(x["stay_id"])
            ids = x["input_ids"]
            tids = x["type_ids"]
            if len(ids) != len(tids):
                type_id_mismatch += 1
            L = len(ids)
            max_len = max(max_len, L)
            if L > 512:
                bad_len += 1

        print(f"\n[{name}] unique stays (sampled): {len(stays)}")
        print(f"[{name}] max seq length (sampled): {max_len}")
        print(f"[{name}] seqs with len>512 (sampled): {bad_len}")
        print(f"[{name}] seqs with len(ids)!=len(type_ids): {type_id_mismatch}")

    # 4. type_ids distribution
    for name, data in [("train", train), ("val", val)]:
        c = Counter()
        for x in data:
            c.update(x["type_ids"])
        print(f"\n[{name}] type_ids counts (sampled):", dict(c))

    # 5. Check eval items: structure + decoding alignment
    tok = AutoTokenizer.from_pretrained(OUT_DIR)
    print(f"\nTokenizer vocab size: {len(tok)}")

    for i in range(5):
        ex = val_eval[i]
        print(f"\n=== val_eval sample {i} ===")
        print("keys:", ex.keys())
        print("stay_id:", ex["stay_id"], "itemid:", ex["itemid"],
              "event_type:", ex["event_type"])
        print("valuenum:", ex["valuenum"])
        print("prompt_len:", len(ex["prompt_ids"]),
              "label_len:", len(ex["label_ids"]))

        print("Prompt tail:")
        print(tok.decode(ex["prompt_ids"][-150:]))

        print("Label text:")
        print(tok.decode(ex["label_ids"]))

    # 6. Check that valuenum roughly matches first number in label_ids
    def extract_first_number(text):
        parts = text.replace("[EOE]", "").split()
        num_str = ""
        for p in parts:
            if p.replace(".", "", 1).replace("-", "", 1).isdigit():
                num_str += p
            elif num_str:
                break
        try:
            return float(num_str)
        except Exception:
            return None

    diffs = []
    for ex in val_eval[:200]:
        lab_text = tok.decode(ex["label_ids"])
        tokens = tok.convert_ids_to_tokens(ex["prompt_ids"][-30:])
        print(tokens)
        pred = extract_first_number(lab_text)
        true = ex["valuenum"]
        if pred is not None:
            diffs.append(abs(pred - true))

    if diffs:
        print(f"\nLabel/valuenum abs diff (median over 200): {sorted(diffs)[len(diffs)//2]:.4f}")
    else:
        print("\nCould not parse any numbers from label_ids; check value formatting.")

    print("\nSanity check complete.")


if __name__ == "__main__":
    sanity_check()
