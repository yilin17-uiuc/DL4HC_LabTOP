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
