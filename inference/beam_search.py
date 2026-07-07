"""
AIRA-LLM
Beam Search Decoding

Deterministic beam search over a trained GPT. Kept simple and readable:
maintains ``beam_width`` candidate sequences ranked by length-normalized
log-probability.
"""

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn.functional as F


@torch.no_grad()
def beam_search(
    model,
    input_ids,
    max_new_tokens: int = 50,
    beam_width: int = 4,
    length_penalty: float = 1.0,
    eos_id: int = None,
    device=None,
):
    """
    input_ids: list[int] prompt tokens.

    Returns: list[int] the best sequence (prompt + generated).
    """

    device = device or next(model.parameters()).device
    model.eval()

    max_len = getattr(model, "max_sequence_length", 256)

    if len(input_ids) == 0:
        input_ids = [0]

    start = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Each beam: (sequence_tensor, cumulative_logprob, finished_flag)
    beams = [(start, 0.0, False)]

    for _ in range(max_new_tokens):

        candidates = []

        for seq, score, finished in beams:

            if finished:
                candidates.append((seq, score, True))
                continue

            logits, _ = model(seq[:, -max_len:])
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

            top_log_probs, top_ids = torch.topk(
                log_probs, beam_width, dim=-1
            )

            for i in range(beam_width):
                token = top_ids[0, i].view(1, 1)
                token_logprob = top_log_probs[0, i].item()

                new_seq = torch.cat([seq, token], dim=1)
                new_score = score + token_logprob
                is_eos = eos_id is not None and token.item() == eos_id

                candidates.append((new_seq, new_score, is_eos))

        # Rank by length-normalized score and keep the best beams.
        def normalized(item):
            seq, score, _ = item
            length = seq.size(1) ** length_penalty
            return score / max(length, 1.0)

        candidates.sort(key=normalized, reverse=True)
        beams = candidates[:beam_width]

        if all(finished for _, _, finished in beams):
            break

    best_seq = max(
        beams,
        key=lambda item: item[1] / max(item[0].size(1) ** length_penalty, 1.0),
    )[0]

    return best_seq[0].tolist()
