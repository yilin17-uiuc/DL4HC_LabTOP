import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizer
from typing import Optional, List, Union, Tuple

# Placeholder for PyHealth BaseModel if not available
try:
    from pyhealth.models import BaseModel
except ImportError:
    class BaseModel(nn.Module):
        """Dummy BaseModel for development environment."""
        def __init__(self, dataset=None, **kwargs):
            super().__init__()
            self.dataset = dataset

# =============================================================================
# Contribution Information
# Name: [Your Name]
# NetId: [Your NetId]
# Paper: LabTOP: Label-Aware Time-Series Pre-training
# Link: [Paper Link]
# Description: Implementation of the LabTOP model (based on GPT-2).
#              It performs autoregressive modeling on medical time-series data.
# =============================================================================


class LabTOPModel(BaseModel):
    """LabTOP Model (Label-Aware Time-Series Pre-training).

    This model uses a GPT-2 architecture to model sequences of medical events.
    It is designed to predict the next token in the sequence, which can be
    used for various downstream tasks such as value prediction or event forecasting.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer used for the dataset.
        d_model (int): Hidden dimension size.
        n_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        max_len (int): Maximum sequence length.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 4,
        max_len: int = 1024,
        dropout: float = 0.1,
        **kwargs
    ):
        super(LabTOPModel, self).__init__(**kwargs)
        self.tokenizer = tokenizer
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.max_len = max_len
        self.dropout = dropout

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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], object]:
        """Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
            attention_mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, seq_len).
            labels (Optional[torch.Tensor]): Target labels for loss calculation.

        Returns:
            outputs: The output from the GPT2LMHeadModel (logits, loss, etc.).
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs

    def generate_next_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 6,
        bad_ids: Optional[List[int]] = None,
        eos_id: Optional[int] = None,
    ) -> List[int]:
        """Generates the next few tokens using greedy decoding.

        Args:
            input_ids (torch.Tensor): Input sequence.
            attention_mask (Optional[torch.Tensor]): Attention mask.
            max_new_tokens (int): Number of tokens to generate.
            bad_ids (Optional[List[int]]): List of token IDs to suppress.
            eos_id (Optional[int]): End-of-sequence token ID to stop generation.

        Returns:
            List[int]: The generated token IDs.
        """
        self.eval()
        generated = []

        with torch.no_grad():
            curr_input_ids = input_ids.clone()
            curr_attention_mask = attention_mask.clone() if attention_mask is not None else None

            for _ in range(max_new_tokens):
                # Truncate if exceeding max length
                if curr_input_ids.size(1) > self.max_len:
                    curr_input_ids = curr_input_ids[:, -self.max_len:]
                    if curr_attention_mask is not None:
                        curr_attention_mask = curr_attention_mask[:, -self.max_len:]

                outputs = self.model(curr_input_ids, attention_mask=curr_attention_mask)
                logits = outputs.logits[:, -1, :]  # (B, vocab)

                if bad_ids:
                    logits[:, bad_ids] = -1e9

                next_token = torch.argmax(logits, dim=-1)  # (B,)
                next_id = next_token.item()
                generated.append(next_id)

                curr_input_ids = torch.cat([curr_input_ids, next_token.unsqueeze(0)], dim=1)
                if curr_attention_mask is not None:
                    next_mask_token = torch.ones_like(next_token).unsqueeze(0)
                    curr_attention_mask = torch.cat([curr_attention_mask, next_mask_token], dim=1)

                if eos_id is not None and next_id == eos_id:
                    break

        return generated

if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    # Add a pad token if missing for the example to work
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = LabTOPModel(tokenizer=tokenizer)
    print("Model initialized successfully.")
