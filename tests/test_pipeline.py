"""
AIRA-LLM
End-to-end and component tests (pytest, assertion based).

These complement the older script-style smoke tests by actually asserting
on shapes, invariants, and a full train -> generate cycle.
"""

import torch

from model.gpt import GPT
from tokenizer.word_tokenizer import WordTokenizer
from tokenizer.char_tokenizer import CharTokenizer
from dataset.preprocess import DatasetPreprocessor
from training.loss import LanguageModelLoss
from inference.temperature import apply_temperature
from inference.topk import top_k_filter
from inference.topp import top_p_filter
from inference.sampler import sample_next_token
from inference.generator import TextGenerator


SAMPLE = "the cat sat on the mat the cat ran to the hat the dog sat on the log"


def tiny_model(vocab_size=64):
    return GPT(
        vocab_size=vocab_size,
        max_sequence_length=32,
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        hidden_dim=64,
        dropout=0.0,
    )


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def test_gpt_forward_shapes():
    model = tiny_model()
    x = torch.randint(0, 64, (2, 16))
    logits, attn = model(x)

    assert logits.shape == (2, 16, 64)
    assert len(attn) == 2
    assert attn[0].shape == (2, 4, 16, 16)


def test_weight_tying():
    model = tiny_model()
    assert model.lm_head.weight is model.token_embedding.embedding.weight


def test_forward_with_padding_mask():
    model = tiny_model()
    x = torch.randint(0, 64, (2, 16))
    mask = torch.ones(2, 16)
    mask[:, 10:] = 0
    logits, _ = model(x, mask)
    assert logits.shape == (2, 16, 64)


def test_causal_attention_is_lower_triangular():
    model = tiny_model()
    x = torch.randint(0, 64, (1, 8))
    _, attn = model(x)
    # Upper triangle (future positions) must be ~0 after softmax.
    weights = attn[0][0, 0]  # (seq, seq)
    upper = torch.triu(weights, diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6)


# ----------------------------------------------------------------------
# Tokenizers
# ----------------------------------------------------------------------
def test_word_tokenizer_roundtrip():
    tok = WordTokenizer()
    tok.train(SAMPLE)
    ids = tok.encode("the cat sat")
    assert all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == "the cat sat"


def test_char_tokenizer_roundtrip():
    tok = CharTokenizer()
    tok.train(SAMPLE)
    ids = tok.encode("the cat")
    assert tok.decode(ids) == "the cat"


# ----------------------------------------------------------------------
# BPE tokenizer (byte-level, no UNK possible)
# ----------------------------------------------------------------------
def test_bpe_roundtrip_seen_and_unseen():
    from tokenizer.bpe_tokenizer import BPETokenizer

    tok = BPETokenizer(vocab_size=400)
    tok.train(SAMPLE * 20)

    # Exact round-trip for seen text, unseen words, and unicode/emoji.
    for s in ["the cat", "zephyr QUANTUM", "你好", "rocket \U0001f680"]:
        assert tok.decode(tok.encode(s)) == s


def test_bpe_never_emits_unk():
    from tokenizer.bpe_tokenizer import BPETokenizer
    from tokenizer.special_tokens import SpecialTokens

    tok = BPETokenizer(vocab_size=400)
    tok.train(SAMPLE * 10)
    # A word never seen in training must still encode without UNK.
    assert SpecialTokens.UNK_ID not in tok.encode("supercalifragilistic")


def test_bpe_save_load(tmp_path):
    from tokenizer.bpe_tokenizer import BPETokenizer

    tok = BPETokenizer(vocab_size=400)
    tok.train(SAMPLE * 10)
    path = tmp_path / "bpe.json"
    tok.save(str(path))

    reloaded = BPETokenizer()
    reloaded.load(str(path))
    assert reloaded.decode(reloaded.encode("the cat")) == "the cat"
    assert len(reloaded) == len(tok)


# ----------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------
def test_loss_is_positive_scalar():
    criterion = LanguageModelLoss()
    logits = torch.randn(2, 10, 50)
    targets = torch.randint(0, 50, (2, 10))
    loss = criterion(logits, targets)
    assert loss.item() > 0
    assert loss.dim() == 0


# ----------------------------------------------------------------------
# Sampling primitives
# ----------------------------------------------------------------------
def test_temperature_scaling():
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    assert torch.allclose(apply_temperature(logits, 1.0), logits)
    assert torch.allclose(apply_temperature(logits, 0.5), logits * 2)


def test_top_k_filter_keeps_k():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    filtered = top_k_filter(logits, k=2)
    assert torch.isinf(filtered).sum().item() == 3  # 5 - 2 masked


def test_top_p_filter_masks_tail():
    logits = torch.tensor([[10.0, 1.0, 1.0, 1.0]])
    filtered = top_p_filter(logits, p=0.5)
    # The dominant token alone exceeds p, so the rest are masked.
    assert torch.isinf(filtered).sum().item() >= 1


def test_sample_next_token_shape():
    logits = torch.randn(3, 50)
    out = sample_next_token(logits, temperature=0.8, top_k=10)
    assert out.shape == (3, 1)


# ----------------------------------------------------------------------
# End-to-end: train a couple of steps, then generate
# ----------------------------------------------------------------------
def test_train_step_reduces_loss():
    torch.manual_seed(0)

    pre = DatasetPreprocessor()
    pre.fit(SAMPLE)
    ids = pre.encode(SAMPLE)
    X, Y = pre.create_sequences(ids, sequence_length=5)

    inputs = torch.tensor(X)
    targets = torch.tensor(Y)

    vocab_size = len(pre.tokenizer.vocab)
    model = tiny_model(vocab_size=vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = LanguageModelLoss()

    logits, _ = model(inputs)
    first_loss = criterion(logits, targets).item()

    for _ in range(50):
        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

    assert loss.item() < first_loss


def test_generator_produces_text():
    pre = DatasetPreprocessor()
    pre.fit(SAMPLE)
    vocab_size = len(pre.tokenizer.vocab)

    model = tiny_model(vocab_size=vocab_size)
    gen = TextGenerator(model, pre.tokenizer, device=torch.device("cpu"))

    out = gen.generate("the cat", max_new_tokens=5, greedy=True)
    assert isinstance(out, str)
    assert len(out) > 0
