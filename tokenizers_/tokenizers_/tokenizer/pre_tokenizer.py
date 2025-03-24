from dataclasses import dataclass
from enum import Enum, auto
from typing import Union, Optional, List, Tuple, Iterable, Any, TypeVar, Callable
from normalizer import NormalizedString, OffsetReferential, Range
from mod import Token

from functools import singledispatch


OffsetType = Enum('OffsetType', ['BYTE', 'CHAR', 'NONE'])
Offsets = Tuple[int, int]


@dataclass
class Split:
    normalized: NormalizedString
    tokens: Optional[List[Token]] = None

    @classmethod
    def from_normalized(cls, n: NormalizedString):
        return cls(normalized=n)

    @classmethod
    def from_tuple(cls, t: Tuple[NormalizedString, Optional[List[Token]]]):
        return cls(normalized=t[0], tokens=t[1])


class PreTokenizedString:
    def __init__(self, s: Union[str, NormalizedString]):
        if isinstance(s, NormalizedString):
            self.original = s.get_original()
            self.splits = [Split(normalized=s)]
        else:
            self.original = s
            self.splits = [Split(normalized=NormalizedString(s))]

    def split(self, split_fn: Callable[[int, NormalizedString], Iterable[Split]]) -> None:
        new_splits = []
        for idx, split in enumerate(self.splits):
            if split.tokens is not None:
                new_splits.append(split)
                continue

            for new_split in split_fn(idx, split.normalized):
                if not new_split.normalized.is_empty():
                    new_splits.append(new_split)
        self.splits = new_splits

    def normalize(self, norm_fn: Callable[[NormalizedString], None]) -> None:
        for split in self.splits:
            if split.tokens is None:
                norm_fn(split.normalized)

    def tokenize(self, tokenize_fn: Callable[[NormalizedString], List[Token]]) -> None:
        for split in self.splits:
            if split.tokens is None:
                split.tokens = tokenize_fn(split.normalized)

    def into_encoding(self, word_idx: Optional[int], type_id: int, offset_type: OffsetType) -> Encoding:
        if not all(split.tokens is not None for split in self.splits):
            raise ValueError("Unprocessed splits")

        converter = BytesToCharOffsetConverter(self.original) if offset_type == OffsetType.CHAR else None

        tokens = []
        for split_idx, split in enumerate(self.splits):
            for token in split.tokens:
                offsets = self._convert_offsets(token.offsets, converter)
                word_id = word_idx if word_idx is not None else split_idx
                tokens.append((token.id, token.value, offsets, word_id, type_id))

        return Encoding(tokens)

    def _convert_offsets(self, offsets: Offsets, converter: Optional['BytesToCharOffsetConverter']) -> Offsets:
        if converter:
            return converter.convert(offsets) or offsets
        return offsets

    def get_splits(self, ref: OffsetReferential, offset_type: OffsetType) -> List[
        Tuple[str, Offsets, Optional[List[Token]]]]:
        converter = BytesToCharOffsetConverter(self.original) if offset_type == OffsetType.CHAR else None
        result = []
        cumulative = 0

        for split in self.splits:
            if ref == OffsetReferential.ORIGINAL:
                offsets = split.normalized.original_offsets()
            else:
                offsets = (cumulative, cumulative + len(split.normalized))
                cumulative += len(split.normalized)

            if converter:
                offsets = converter.convert(offsets) or offsets

            result.append((split.normalized.get(), offsets, split.tokens))

        return result


class BytesToCharOffsetConverter:
    def __init__(self, s: str):
        self.mapping = {}
        char_idx = 0
        for byte_idx, c in enumerate(s.encode('utf-8')):
            self.mapping[byte_idx] = char_idx
            if c & 0b11000000 != 0b10000000:  # 判断是否为新字符起始
                char_idx += 1

    def convert(self, offsets: Offsets) -> Optional[Offsets]:
        start = self.mapping.get(offsets[0])
        end = self.mapping.get(offsets[1] - 1, start)
        return (start, end + 1) if start is not None and end is not None else None


# 示例用法 ---------------
if __name__ == "__main__":
    # 假设 NormalizedString 的实现
    ns = NormalizedString("Hello world")
    pts = PreTokenizedString(ns)


    # 定义拆分函数
    def split_func(idx: int, ns: NormalizedString) -> List[Split]:
        return [Split.from_normalized(ns.split()[0])]


    pts.split(split_func)
    pts.tokenize(lambda ns: [Token(0, "hello", (0, 5))])
    encoding = pts.into_encoding(None, 0, OffsetType.CHAR)

