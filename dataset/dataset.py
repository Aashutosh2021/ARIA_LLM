"""
AIRA-LLM
PyTorch Dataset
"""

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset


class LLMDataset(Dataset):

    def __init__(self, inputs, targets):

        self.inputs = inputs
        self.targets = targets

    # ----------------------------
    # Dataset Length
    # ----------------------------
    def __len__(self):

        return len(self.inputs)

    # ----------------------------
    # Get Item
    # ----------------------------
    def __getitem__(self, index):

        x = torch.tensor(
            self.inputs[index],
            dtype=torch.long
        )

        y = torch.tensor(
            self.targets[index],
            dtype=torch.long
        )

        return x, y