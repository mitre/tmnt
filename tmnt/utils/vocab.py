import torch
import torch.nn as nn
from typing import Dict, List, Optional, Iterable, OrderedDict
from collections import Counter

class Vocab(nn.Module):
    r"""Creates a vocab object which maps tokens to indices.

    Args:
        vocab (torch.classes.torchtext.Vocab or torchtext._torchtext.Vocab): a cpp vocab object.
    """

    def __init__(self, stoi: Dict):
        super(Vocab, self).__init__()
        self.stoi = stoi
        self.itos = list(stoi.keys())

    def forward(self, tokens: List[str]) -> List[int]:
        r"""Calls the `lookup_indices` method

        Args:
            tokens: a list of tokens used to lookup their corresponding `indices`.

        Returns:
            The indices associated with a list of `tokens`.
        """
        return [self.stoi[t] for t in tokens]

    def __len__(self) -> int:
        r"""
        Returns:
            The length of the vocab.
        """
        return len(self.stoi)

    def __contains__(self, token: str) -> bool:
        r"""
        Args:
            token: The token for which to check the membership.

        Returns:
            Whether the token is member of vocab or not.
        """
        return self.stoi.__contains__(token)

    def __getitem__(self, token: str) -> int:
        r"""
        Args:
            token: The token used to lookup the corresponding index.

        Returns:
            The index corresponding to the associated token.
        """
        return self.stoi[token]

    def insert_token(self, token: str, index: int) -> None:
        r"""
        Args:
            token: The token used to lookup the corresponding index.
            index: The index corresponding to the associated token.
        Raises:
            RuntimeError: If `index` is not in range [0, Vocab.size()] or if `token` already exists in the vocab.
        """
        if not token in self.stoi:
            self.stoi[token] = index
            self.itos[index] = token

    def lookup_token(self, index: int) -> str:
        r"""
        Args:
            index: The index corresponding to the associated token.

        Returns:
            token: The token used to lookup the corresponding index.

        Raises:
            RuntimeError: If `index` not in range [0, itos.size()).
        """
        return self.itos[index]

    def lookup_tokens(self, indices: List[int]) -> List[str]:
        r"""
        Args:
            indices: The `indices` used to lookup their corresponding`tokens`.

        Returns:
            The `tokens` associated with `indices`.

        Raises:
            RuntimeError: If an index within `indices` is not int range [0, itos.size()).
        """
        return [ self.itos[i] for i in indices]

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        r"""
        Args:
            tokens: the tokens used to lookup their corresponding `indices`.

        Returns:
            The 'indices` associated with `tokens`.
        """
        return [ self.stoi[t] for t in tokens ]

    def get_stoi(self) -> Dict[str, int]:
        r"""
        Returns:
            Dictionary mapping tokens to indices.
        """
        return self.stoi

    def get_itos(self) -> List[str]:
        r"""
        Returns:
            List mapping indices to tokens.
        """
        return self.itos



def build_vocab(
    odict: OrderedDict
) -> Vocab:
    """
    """
    dict_by_position = dict(zip(odict.keys(), range(0,len(odict))))
    return Vocab(dict_by_position)