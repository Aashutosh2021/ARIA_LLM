"""
AIRA-LLM
Special Tokens

These IDs are RESERVED.
Never change them.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialTokens:
    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    MASK = "<MASK>"

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    MASK_ID = 4


SPECIAL_TOKEN_TO_ID = {
    SpecialTokens.PAD: SpecialTokens.PAD_ID,
    SpecialTokens.UNK: SpecialTokens.UNK_ID,
    SpecialTokens.BOS: SpecialTokens.BOS_ID,
    SpecialTokens.EOS: SpecialTokens.EOS_ID,
    SpecialTokens.MASK: SpecialTokens.MASK_ID,
}


SPECIAL_ID_TO_TOKEN = {
    v: k for k, v in SPECIAL_TOKEN_TO_ID.items()
}


VOCAB_START_INDEX = len(SPECIAL_TOKEN_TO_ID)

print(SPECIAL_TOKEN_TO_ID)
print(SPECIAL_ID_TO_TOKEN)
print(VOCAB_START_INDEX)