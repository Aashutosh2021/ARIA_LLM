"""
AIRA-LLM
Text Generator

Autoregressive decoding loop around a trained GPT model. Supports
greedy decoding and temperature / top-k / top-p sampling. Handles the
context window by cropping to the model's ``max_sequence_length``.
"""

# pyrefly: ignore [missing-import]
import torch

from inference.sampler import sample_next_token


class TextGenerator:

    def __init__(self, model, tokenizer, device=None):

        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device

        self.model.to(self.device)
        self.model.eval()

        # Context window; fall back to a sane default if unset.
        self.max_len = getattr(model, "max_sequence_length", 256)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt: str = "",
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        greedy: bool = False,
        eos_id: int = None,
    ) -> str:
        """Generate text continuing from ``prompt`` and return the string."""

        ids = self.tokenizer.encode(prompt) if prompt else []

        generated = self._generate_ids(
            ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
            eos_id=eos_id,
        )

        return self.tokenizer.decode(generated)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate_ids(
        self,
        ids,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        greedy,
        eos_id,
    ):

        # Seed with a single BOS-like token if the prompt is empty so the
        # model has something to condition on.
        if len(ids) == 0:
            ids = [0]

        tokens = torch.tensor(
            [ids],
            dtype=torch.long,
            device=self.device,
        )

        for _ in range(max_new_tokens):

            # Crop context to the model's max sequence length.
            context = tokens[:, -self.max_len:]

            logits, _ = self.model(context)

            next_logits = logits[:, -1, :]

            next_token = sample_next_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                greedy=greedy,
            )

            tokens = torch.cat([tokens, next_token], dim=1)

            if eos_id is not None and next_token.item() == eos_id:
                break

        return tokens[0].tolist()
