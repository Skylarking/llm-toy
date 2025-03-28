from typing import List, Tuple
from ..tokenizer.normalizer import NormalizedString
from .base import Normalizer
class Sequence(Normalizer):
    def __init__(self, normalizers: List[Normalizer]):
        self.normalizers = normalizers

    def normalize(self, normalized: NormalizedString):
        for normalizer in self.normalizers:
            normalizer.normalize(normalized)

    def normalize_str(self, sequence: str):
        # 创建临时对象进行流水线处理
        nstr = NormalizedString(sequence)
        self.normalize(nstr)
        return nstr.get()


class Lowercase(Normalizer):
    def normalize(self, normalized: NormalizedString):
        normalized.lowercase()

    def normalize_str(self, sequence: str):
        nstr = NormalizedString(sequence)
        nstr.lowercase()
        return nstr.get()

if __name__ == '__main__':
    from unicode import NFD
    from strip import StripAccents
    normalizer = Sequence([NFD(), StripAccents(), Lowercase()])
    ns = NormalizedString("Héllò hôw are ü?")
    normalizer.normalize(ns)
    print(f"str: {ns.get()}, alignments: {ns.alignments}")
    print(len(ns.get()), len(ns.alignments))