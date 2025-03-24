from typing import List, Tuple
from tokenizers_.tokenizer import NormalizedString
from tokenizers_.normalizers.base import Normalizer
import unicodedata
class Strip(Normalizer):
    strip_left: bool
    strip_right: bool

    def __init__(self,
                 strip_left: bool = True,
                 strip_right: bool = True):
        self.strip_left = strip_left
        self.strip_right = strip_right

    def normalize(self, normalized: NormalizedString):
        if self.strip_left and self.strip_right:
            normalized.strip()
        else:
            if self.strip_left:
                normalized.lstrip()
            if self.strip_right:
                normalized.rstrip()

    def normalize_str(self, sequence: str):
        nstr = NormalizedString(sequence)
        if self.strip_left and self.strip_right:
            nstr.strip()
        else:
            if self.strip_left:
                nstr.lstrip()
            if self.strip_right:
                nstr.rstrip()
        return nstr.get()


class StripAccents(Normalizer):
    def normalize(self, normalized: NormalizedString):
        normalized.filter(lambda c: not unicodedata.combining(c))

    def normalize_str(self, sequence: str):
        nstr = NormalizedString(sequence)
        nstr.filter(lambda c: not unicodedata.combining(c))
        return nstr.get()