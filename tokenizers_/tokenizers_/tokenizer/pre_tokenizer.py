from dataclasses import dataclass
from enum import Enum, auto
from typing import Union, Optional, List, Tuple, Iterable, Any, TypeVar, Callable
from .normalizer import NormalizedString, OffsetReferential, Range, SplitDelimiterBehavior
from .mod import Token, Offsets
from .encoding import Encoding

class OffsetType(Enum):
    BYTE = auto()
    CHAR = auto()
    NONE = auto()


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

    def is_empty(self) -> bool:
        return self.normalized.is_empty()


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

            try:
                # 增加空值过滤和异常处理
                for new_split in split_fn(idx, split.normalized):
                    if not new_split.is_empty():
                        new_splits.append(new_split)
            except Exception as e:
                raise RuntimeError(f"Split failed at index {idx}: {str(e)}")
        self.splits = new_splits

    def normalize(self, norm_fn: Callable[[NormalizedString], None]) -> None:
        for split in self.splits:
            if split.tokens is None:
                norm_fn(split.normalized)

    def tokenize(self, tokenize_fn: Callable[[NormalizedString], List[Token]]) -> None:
        for split in self.splits:
            if split.tokens is None:
                # 使用工厂方法创建Token
                split.tokens = [
                    Token.new(
                        id=token.id,
                        value=token.value,
                        offsets=token.offsets
                    ) for token in tokenize_fn(split.normalized)
                ]

    def into_encoding(
            self,
            word_idx: Optional[int],
            type_id: int,
            offset_type: OffsetType
    ) -> Encoding:
        if not all(split.tokens is not None for split in self.splits):
            raise ValueError("存在未处理的splits，请先调用tokenize方法")

        converter = BytesToCharOffsetConverter(self.original) if offset_type == OffsetType.CHAR else None

        # 创建新的Token列表（关键修正点）
        converted_tokens = []
        for split in self.splits:
            for token in split.tokens:
                # 创建新Token实例而不是修改原实例
                new_offsets = token.offsets
                if converter:
                    new_offsets = converter.convert(token.offsets) or token.offsets

                converted_tokens.append(
                    Token.new(
                        id=token.id,
                        value=token.value,
                        offsets=new_offsets  # 使用转换后的偏移量
                    )
                )

        # 使用转换后的Token列表构建Encoding
        encoding = Encoding.from_tokens(converted_tokens, type_id)

        # 处理word_idx参数（如果需要）
        if word_idx is not None:
            encoding.words = [word_idx] * len(converted_tokens)

        return encoding

    def _safe_convert_offsets(self, offsets: Offsets, converter: Optional['BytesToCharOffsetConverter']) -> Offsets:
        """带边界检查的偏移量转换"""
        if converter is None:
            return offsets

        converted = converter.convert(offsets)
        if converted is None:
            raise ValueError(f"Invalid offsets conversion: {offsets}")
        return converted

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
            # 获取正确的偏移量基准
            if ref == OffsetReferential.ORIGINAL:
                base_offsets = split.normalized.offsets_original()
            else:
                base_offsets = (cumulative, cumulative + len(split.normalized))
                cumulative += len(split.normalized)

            # 安全转换
            final_offsets = self._safe_convert_offsets(base_offsets, converter)

            result.append((
                split.normalized.get(),
                final_offsets,
                split.tokens
            ))

        return result


class BytesToCharOffsetConverter:
    def __init__(self, s: str):
        self.mapping = {}
        char_idx = 0
        byte_pos = 0

        # 生成完整的字节到字符映射
        for c in s:
            char_bytes = c.encode('utf-8')
            for _ in char_bytes:
                self.mapping[byte_pos] = char_idx
                byte_pos += 1
            char_idx += 1

        # 添加哨兵值处理字符串末尾
        self.mapping[byte_pos] = char_idx

    def convert(self, offsets: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """"""
        start_byte, end_byte = offsets
        # 获取起始字符索引
        start_char = self.mapping.get(start_byte)
        # 获取结束字符索引（无需+1）
        end_char = self.mapping.get(end_byte)

        if start_char is not None and end_char is not None:
            return (start_char, end_char)
        return None


# 示例用法 ---------------
if __name__ == "__main__":
    ns = NormalizedString("Hello world")
    pts = PreTokenizedString(ns)

    # 定义拆分函数（使用正确的SplitDelimiterBehavior）
    def split_func(idx: int, ns: NormalizedString) -> List[Split]:
        return [Split.from_normalized(ns.split(' ', SplitDelimiterBehavior.REMOVED)[0])]

    # 定义tokenize函数
    def tokenize_func(ns: NormalizedString) -> List[Token]:
        return [
            Token.new(0, "Hello", (0, 5)),
            Token.new(1, "world", (6, 11))
        ]

    # 处理流程
    pts.split(split_func)
    pts.tokenize(tokenize_func)

    # 生成字符级偏移编码
    encoding = pts.into_encoding(
        word_idx=None,
        type_id=0,
        offset_type=OffsetType.CHAR
    )

    # 验证结果
    print(encoding.offsets)  # 应输出字符级偏移如 [(0,5), (6,11)]
    print(encoding.tokens)  # ["Hello", "world"]

    # 中文测试
    print("+----------------- 中文测试 ----------------+")
    ns = NormalizedString("中文测试")
    pts = PreTokenizedString(ns)


    # 定义拆分逻辑
    def split_func(idx: int, ns: NormalizedString) -> List[Split]:
        return [Split.from_normalized(ns)]


    # 定义tokenize逻辑
    def tokenize_func(ns: NormalizedString) -> List[Token]:
        return [
            Token.new(0, "中", (0, 3)),
            Token.new(1, "文", (3, 6)),
            Token.new(2, "测", (6, 9)),
            Token.new(3, "试", (9, 12))
        ]


    # 处理流程
    pts.split(split_func)
    pts.tokenize(tokenize_func)

    # 生成编码
    encoding = pts.into_encoding(
        word_idx=None,  # 该参数已废弃
        type_id=0,
        offset_type=OffsetType.CHAR
    )

    # 验证结果
    assert encoding.ids == [0, 1, 2, 3]
    assert encoding.offsets == [(0, 1), (1, 2), (2, 3), (3, 4)]  # 字符级偏移
    assert encoding.type_ids == [0, 0, 0, 0]

