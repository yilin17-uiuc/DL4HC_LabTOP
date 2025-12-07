import torch, pickle, os
from transformers import AutoTokenizer
from types import SimpleNamespace

data_dir = "new_processed_data"
tokenizer = AutoTokenizer.from_pretrained(data_dir)
vocab_size = len(tokenizer)
ckpt_path = os.path.join(data_dir, "labtop_model.pth")

# import your model class (adjust import if needed)
from evaluate import LabTOPModel   # or from your evaluate.py

model = LabTOPModel(vocab_size=vocab_size).to("cpu")
sd = torch.load(ckpt_path, map_location="cpu")
res = model.load_state_dict(sd, strict=False)
print("load_state_dict returned:", res)

# Forward one example
with open(os.path.join(data_dir, "test_eval.pkl"), "rb") as f:
    test_data = pickle.load(f)
example = test_data[0]["prompt_ids"]
input_ids = torch.tensor([example], dtype=torch.long)
out = model(input_ids)
print("type(out):", type(out))
# if it's a tensor show shape, else show whether .logits exists
if isinstance(out, torch.Tensor):
    print("out is Tensor shape:", out.shape)
else:
    print("has logits:", hasattr(out, "logits"), "logits shape:", out.logits.shape if hasattr(out, "logits") else None)


# compute raw logits (no softmax) for last position for first example
model.eval()
with torch.no_grad():
    out = model(input_ids)
    logits = out.logits if not isinstance(out, torch.Tensor) else out
    last_logits = logits[0, -1]   # shape: (vocab,)
# show top 12 logits (raw) and their decodes
topk = torch.topk(last_logits, 12)
ids = topk.indices.tolist()
vals = topk.values.tolist()
print("Top-12 raw logits (id,logit,token):")
for i,v in zip(ids, vals):
    print(i, v, tokenizer.decode([int(i)]))

# inspect lm_head bias & tok_emb norm
lm_bias = None
try:
    lm_bias = model.lm_head.bias.detach().cpu().numpy()
    import numpy as np
    print("lm_head.bias: min,mean,max:", float(lm_bias.min()), float(lm_bias.mean()), float(lm_bias.max()))
except Exception as e:
    print("can't read lm_head.bias:", e)

try:
    emb = model.tok_emb.weight.detach().cpu()
    norms = emb.norm(dim=1)
    print("tok_emb norms: min,mean,max:", float(norms.min()), float(norms.mean()), float(norms.max()))
except Exception as e:
    print("can't read tok_emb:", e)


with torch.no_grad():
    if hasattr(model.lm_head, "bias"):
        model.lm_head.bias.zero_()
        print("Zeroed lm_head.bias")
    else:
        print("no lm_head.bias found")
# now sample greedily 10 tokens for the example (same loop you had but simplified)
model.eval()
generated = []
curr = torch.tensor([example], dtype=torch.long)
for _ in range(10):
    with torch.no_grad():
        out = model(curr)
        logits = out.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1)
        generated.append(int(next_token.item()))
        curr = torch.cat([curr, next_token.unsqueeze(0)], dim=1)
print("After zeroing bias -> decoded:", tokenizer.decode(generated))

with torch.no_grad():
    model.lm_head.weight.data = model.tok_emb.weight.data
    if hasattr(model.lm_head, "bias"):
        model.lm_head.bias.zero_()
    print("Tied lm_head weight to tok_emb and zeroed bias")
# sample again same as above
model.eval()
generated = []
curr = torch.tensor([example], dtype=torch.long)
for _ in range(10):
    with torch.no_grad():
        out = model(curr)
        logits = out.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1)
        generated.append(int(next_token.item()))
        curr = torch.cat([curr, next_token.unsqueeze(0)], dim=1)
print("After tying weights -> decoded:", tokenizer.decode(generated))

