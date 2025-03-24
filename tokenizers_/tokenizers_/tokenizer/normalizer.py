from dataclasses import dataclass
from enum import Enum, auto
from typing import Union, Optional, List, Tuple, Iterable, Any, TypeVar, Callable
from unicodedata import normalize
import re
from difflib import SequenceMatcher

# 范型注解，用于Range中bounds，可以有多种类型
T = TypeVar('T')


class OffsetReferential(Enum):
    ORIGINAL = auto()   # 原始str的偏移参考系
    NORMALIZED = auto() # norm str的偏移参考系
# range_orig = Range.Original(slice(3, 5))  # 基于原始字符串的3-5字符范围
# range_norm = Range.Normalized(range(1, 4))  # 基于规范化字符串的1-4字符范围


class Range:
    '''
    统一处理不同参考系的查询范围
    通过referential明确所属参考系

    '''
    def __init__(self, referential: OffsetReferential, bounds: Union[range, slice]):
        self.referential = referential
        self.bounds = bounds

    @classmethod
    def Original(cls, bounds: Union[range, slice]):
        return cls(OffsetReferential.ORIGINAL, bounds)

    @classmethod
    def Normalized(cls, bounds: Union[range, slice]):
        return cls(OffsetReferential.NORMALIZED, bounds)


#
class SplitDelimiterBehavior(Enum):
    REMOVED = auto()        # 删除分隔符  "a-b" ->  ["a", "b"]
    ISOLATED = auto()       # 分隔符作为独立元素   "a-b" ->  ["a", "-", "b"]
    MERGED_WITH_PREVIOUS = auto()   # 分隔符合并到前段  "a-b" ->  ["a-", "b"]
    MERGED_WITH_NEXT = auto()   # 分隔符合并到后段  "a-b" ->  ["a", "-b"]
    CONTIGUOUS = auto()         # 连续分隔符视为整体 "a--b" ->  ["a", "--", "b"]


@dataclass
class NormalizedString:
    """
    alignments:
        核心作用：
            记录规范化字符串每个字节对应原始字符串的字节范围，实现双向偏移转换。
        技术细节：
            结构：列表中的每个元素是二元组 (start, end)
            索引：每个元素对应规范化字符串的一个字节（不是字符）
            数值：
                start：该字节在原始字符串中字符的起始字节位置
                end：该字节在原始字符串中字符的结束字节位置
        示例说明：
            原始字符串："café"（字节表示：b'caf\xc3\xa9'，共5字节）
            规范化操作：NFD分解后得到"cafe\u0301"（字节表示：b'cafe\xcc\x81'，共6字节）
            此时alignments的结构：
                [
                    (0,1),  # 'c' -> 原始0字节
                    (1,2),  # 'a' -> 原始1字节
                    (2,3),  # 'f' -> 原始2字节
                    (3,5),  # 'e' -> 原始3-5字节
                    (3,5),  # 组合符号́ -> 原始3-5字节
                    (3,5)   # 组合符号́ -> 原始3-5字节
                ]
    original_shift:
        核心作用：
            跟踪当前字符串在原始母字符串中的位置偏移，用于处理字符串切片时的坐标转换。
        技术细节：
            初始值：0（表示完整字符串）
            累计规则：当进行切片操作时，该值会累加前一个切片的偏移量
            计算原理：
                子字符串的original_shift = 父字符串的original_shift + 子字符串在父字符串中的起始位置
        示例说明：
            原始母字符串："Hello World"（长度11字节）
                ```python
                parent = NormalizedString("Hello World")  # original_shift=0

                # 截取第6-11字节（"World"）
                child = parent.slice(Range.Original(6..11))

                # 子字符串的original_shift=6
                # 因为"World"在母字符串中起始于第6字节
                ```
        典型应用场景：
            ```python
            parent = NormalizedString("abcdefghijk")  # original_shift=0

            child1 = parent.slice(Range.Original(2..5))  # original_shift=2
            child2 = child1.slice(Range.Original(1..3))  # original_shift=2+1=3

            # 当查询child2的某个位置时：
            child2.convert_offsets(Range.Normalized(0..1))
            # 实际计算时会加上original_shift=3，得到母字符串的位置
            ```
    """
    original: str
    normalized: str
    alignments: List[Tuple[int, int]]   # index表示原始字符的一个字节的下标，元素是tuple[int, int]，第一个int表示norm后的开始位置，第二个int是norm后的结束位置的后一个位置（左闭右开）
    original_shift: int = 0

    def __init__(self, sequence: str):
        self.original = sequence
        self.normalized = sequence
        self.alignments = []
        byte_pos = 0
        for char in sequence:
            char_bytes = char.encode('utf-8')
            char_bytes_len = len(char_bytes)
            for _ in char_bytes:
                self.alignments.append((byte_pos, byte_pos + char_bytes_len))
            byte_pos += char_bytes_len
        self.original_shift = 0

    # region Core Methods
    def get(self) -> str:
        return self.normalized

    def get_original(self) -> str:
        return self.original

    @classmethod
    def set(cls,
            sequence: str,
            normalized: str,
            alignments: List[Tuple[int, int]],
            original_shift: int = 0) -> 'NormalizedString':
        new = cls(sequence)
        new.normalized = normalized
        new.alignments = alignments
        new.original_shift = original_shift
        return new


    # endregion

    # region Transformation Methods
    def _transform(self, transitions: Iterable[tuple]) -> None:
        """执行转换并更新对齐信息"""
        new_norm = []
        new_align = []
        current_byte = 0  # 当前处理的原始字节位置

        for item in transitions:
            if item[1] == 'keep':
                # 保留字符：直接复制原始对齐信息
                for c in item[0]:
                    char_bytes = c.encode('utf-8')
                    for _ in char_bytes:
                        new_align.append(self.alignments[current_byte])
                    new_norm.append(c)
                    current_byte += len(char_bytes)

            elif item[1] == 'replace':
                # 替换操作：新字符继承被替换字符的原始字节块
                new_chars, _, old_chars = item
                # 计算被替换字符的总字节数
                replaced_bytes = sum(len(c.encode()) for c in old_chars)
                # 新字符的每个字节继承整个被替换块的原始范围
                for c in new_chars:
                    char_bytes = c.encode('utf-8')
                    for _ in char_bytes:
                        new_align.append((current_byte, current_byte + replaced_bytes))
                    new_norm.append(c)
                current_byte += replaced_bytes

            elif item[1] == 'insert':
                # 插入新字符：对齐到前一个字符的结束位置
                new_chars = item[0]
                last_end = new_align[-1][1] if new_align else current_byte
                for c in new_chars:
                    char_bytes = c.encode('utf-8')
                    for _ in char_bytes:
                        new_align.append((last_end, last_end))
                    new_norm.append(c)

            elif item[1] == 'delete':
                # 删除操作：直接跳过原始字节
                deleted_chars = item[0]
                current_byte += sum(len(c.encode()) for c in deleted_chars)

        self.normalized = ''.join(new_norm)
        self.alignments = new_align


    def nfd(self) -> 'NormalizedString':
        return self._apply_unicode_normalization('NFD')

    def nfkd(self) -> 'NormalizedString':
        return self._apply_unicode_normalization('NFKD')

    def nfc(self) -> 'NormalizedString':
        return self._apply_unicode_normalization('NFC')

    def nfkc(self) -> 'NormalizedString':
        return self._apply_unicode_normalization('NFKC')

    def _apply_unicode_normalization(self, form: str) -> 'NormalizedString':
        """应用Unicode规范化并维护精确的字节级对齐"""
        original = self.normalized
        normalized = normalize(form, original)

        # 生成字符级操作序列（基于difflib）
        matcher = SequenceMatcher(None, original, normalized)
        transitions = []

        # 第一步：生成字符级操作指令
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # 保留原始字符，每个字符映射到其原始字节块
                for c in original[i1:i2]:
                    transitions.append((c, 'keep'))
            elif tag == 'replace':
                # 替换操作：新字符继承被替换字符的原始字节块
                old_chars = original[i1:i2]
                new_chars = normalized[j1:j2]
                transitions.append((new_chars, 'replace', old_chars))
            elif tag == 'insert':
                # 插入新字符，不继承任何原始字节块
                transitions.append((normalized[j1:j2], 'insert'))
            elif tag == 'delete':
                # 删除操作：标记被删除的原始字节块
                transitions.append((original[i1:i2], 'delete'))

        # 第二步：转换操作序列到字节级
        self._transform(transitions)
        return self

    # endregion

    # region 字符串操作
    def is_char_boundary(self, index: int) -> bool:
        """检查字节索引是否是字符的起始位置（仿照 Rust 的 is_char_boundary）"""
        if index == 0 or index == len(self.normalized.encode('utf-8')):
            return True
        try:
            # 检查该字节是否是一个字符的起始字节
            self.normalized.encode()[:index].decode()
            return True
        except UnicodeDecodeError:
            return False

    def slice(self, rng: Range) -> Optional['NormalizedString']:
        """安全切片（严格对齐字符边界）"""
        converted = self.convert_offsets(rng)
        if not converted:
            return None

        start, end = converted.start, converted.stop

        # 严格检查字节边界（HF tokenizers 核心约束）
        if not (self.is_char_boundary(start) and self.is_char_boundary(end)):
            return None

        # 提取子字符串和对齐信息
        new_norm = self.normalized.encode('utf-8')[start:end].decode('utf-8')
        new_align = self.alignments[start:end]

        return NormalizedString.set(
            self.original,
            normalized=new_norm,
            alignments=new_align,
            original_shift=self.original_shift + self._get_original_start(start)
        )

    def convert_offsets(self, rng: Range) -> Optional[range]:
        """符合 HF tokenizers 规范的偏移转换"""

        def _resolve_bounds(bounds: Union[range, slice], max_len: int) -> Tuple[int, int]:
            # 统一处理 range/slice 的边界解析
            if isinstance(bounds, slice):
                start = bounds.start if bounds.start is not None else 0
                stop = bounds.stop if bounds.stop is not None else max_len
            elif isinstance(bounds, range):
                if bounds.step != 1:
                    raise ValueError("Only step=1 ranges are supported")
                start, stop = bounds.start, bounds.stop
            else:
                raise TypeError("Unsupported bounds type")

            # 处理负数索引
            start = max(0, start + max_len) if start < 0 else min(start, max_len)
            stop = max(0, stop + max_len) if stop < 0 else min(stop, max_len)
            return start, stop

        # 分参考系处理
        if rng.referential == OffsetReferential.ORIGINAL:
            # 原始参考系转换需要严格字符对齐
            original_start, original_end = _resolve_bounds(rng.bounds, len(self.original.encode()))

            # 收集完全包含在原始范围内的字符
            norm_indices = []
            current_byte = 0
            for i, char in enumerate(self.normalized):
                char_bytes = char.encode('utf-8')
                char_len = len(char_bytes)
                align = self.alignments[current_byte]
                # 检查当前字符的原始范围是否完全在目标范围内
                if align[0] >= original_start and align[1] <= original_end:
                    norm_indices.extend(range(current_byte, current_byte + char_len))
                current_byte += char_len

            if not norm_indices:
                return None
            return range(norm_indices[0], norm_indices[-1] + 1)

        elif rng.referential == OffsetReferential.NORMALIZED:
            # 规范化参考系直接截取，但后续会检查字符边界
            max_len = len(self.alignments)
            start, stop = _resolve_bounds(rng.bounds, max_len)
            return range(start, stop) if start <= stop else None

        else:
            raise ValueError("Invalid referential")

    def _get_original_start(self, norm_byte: int) -> int:
        """获取原始字符串的起始字节偏移"""
        return self.alignments[norm_byte][0] if self.alignments else 0

    def filter(self, predicate: Callable[[str], bool]) -> 'NormalizedString':
        """过滤字符"""
        transitions = []

        # 遍历规范化字符串的每个字符（字符级处理）
        current_byte = 0
        for char in self.normalized:
            # 获取当前字符UTF-8字节长度
            char_byte_len = len(char.encode('utf-8'))

            # 检查当前字节是否属于多字节字符的中间部分
            if current_byte + char_byte_len > len(self.alignments):
                break  # 防止越界

            # 判断是否保留该字符
            if predicate(char):
                # 生成 'keep' 操作：保留字符及其原始字节块
                transitions.append(([char], 'keep', []))
            else:
                # 生成 'delete' 操作：标记删除的字符
                transitions.append(([char], 'delete'))

            current_byte += char_byte_len

        self._transform(transitions)
        return self

    def map(self, func: Callable[[str], str]) -> 'NormalizedString':
        """按字符处理，确保 current_byte 正确递增"""
        transitions = []
        current_byte = 0  # 当前处理的字节位置（基于字节索引）

        # 遍历规范化字符串的每个字符（字符级循环）
        for char in self.normalized:
            # 获取当前字符的 UTF-8 字节长度
            char_byte_len = len(char.encode('utf-8'))

            # 检查是否越界（确保字符的字节范围在 alignments 内）
            if current_byte + char_byte_len > len(self.alignments):
                break

            # 应用映射函数生成新字符
            new_char = func(char)

            # 生成替换操作指令：(new_char, 'replace', [old_char])
            transitions.append(
                (new_char, 'replace', [char])
            )

            # 更新 current_byte 到下一个字符的起始字节位置
            current_byte += char_byte_len

        self._transform(transitions)
        return self

    def lowercase(self) -> 'NormalizedString':
        """小写化并维护对齐"""
        return self.map(lambda c: c.lower())

    def uppercase(self) -> 'NormalizedString':
        """大写化并维护对齐"""
        return self.map(lambda c: c.upper())

    def lstrip(self):
        """只去除左侧空白"""
        whitespace = {' ', '\t', '\n', '\r', '\v', '\f'}
        # 定位左边界
        left = 0
        while left < len(self.normalized) and self.normalized[left] in whitespace:
            left += 1

        # 构建操作序列：删除左侧 + 保留右侧
        transitions = [
            *[([self.normalized[i]], 'delete') for i in range(left)],
            *[([self.normalized[i]], 'keep', []) for i in range(left, len(self.normalized))]
        ]

        # 应用变换
        self._transform(transitions)

        # 更新偏移量（第一个保留字符的原始位置）
        if left < len(self.alignments):
            self.original_shift = self.alignments[left][0] if left < len(self.alignments) else 0
        return self

    def rstrip(self):
        """只去除右侧空白"""
        whitespace = {' ', '\t', '\n', '\r', '\v', '\f'}
        # 定位右边界
        right = len(self.normalized) - 1
        while right >= 0 and self.normalized[right] in whitespace:
            right -= 1

        # 构建操作序列：保留左侧 + 删除右侧
        transitions = [
            *[([self.normalized[i]], 'keep', []) for i in range(0, right + 1)],
            *[([self.normalized[i]], 'delete') for i in range(right + 1, len(self.normalized))]
        ]

        # 应用变换
        self._transform(transitions)
        # 右侧去除不需要更新偏移量
        return self

    def strip(self):
        whitespace = {' ', '\t', '\n', '\r', '\v', '\f'}
        # 阶段1：确定左右边界（与filter的predicate逻辑解耦）
        left = 0
        while left < len(self.normalized) and self.normalized[left] in whitespace:
            left += 1

        right = len(self.normalized) - 1
        while right >= 0 and self.normalized[right] in whitespace:
            right -= 1

        # 阶段2：构建三段式操作序列（left前删除，中间保留，right后删除）
        transitions = []
        # 左侧删除操作
        for i in range(left):
            transitions.append(([self.normalized[i]], 'delete'))
        # 中间保留操作
        for i in range(left, right + 1):
            transitions.append(([self.normalized[i]], 'keep', []))
        # 右侧删除操作
        for i in range(right + 1, len(self.normalized)):
            transitions.append(([self.normalized[i]], 'delete'))

        # 阶段3：应用变换（仿照filter的_transform调用）
        self._transform(transitions)
        # 计算偏移量（首个保留字符的原始字节起始位置）
        self.original_shift = self.alignments[left][0] if left <= right else 0
        return self

    # endregion

    # region 辅助方法
    # def convert_offsets(self, rng: Range) -> Optional[range]:
    #     """
    #     将原始字符串或规范化字符串的字节范围转换为规范化字符串的字节索引范围
    #     Args:
    #         rng (Range): 包含参考系（ORIGINAL/NORMALIZED）和范围（range/slice）
    #     Returns:
    #         Optional[range]: 规范化字符串的连续字节范围，无法转换时返回None
    #     """
    #
    #     # 辅助函数：解析范围边界
    #     def _resolve_bounds(bounds: Union[range, slice], total_length: int) -> Tuple[int, int]:
    #         if isinstance(bounds, slice):
    #             start = bounds.start if bounds.start is not None else 0
    #             stop = bounds.stop if bounds.stop is not None else total_length
    #             # 处理负数索引
    #             start = max(0, start + total_length) if start < 0 else start
    #             stop = max(0, stop + total_length) if stop < 0 else stop
    #             return (max(0, min(start, total_length)), max(0, min(stop, total_length)))
    #         elif isinstance(bounds, range):
    #             if bounds.step != 1:
    #                 raise ValueError("Only step=1 is supported for range bounds.")
    #             start = max(0, min(bounds.start, total_length))
    #             stop = max(0, min(bounds.stop, total_length))
    #             return (start, stop)
    #         else:
    #             raise TypeError("Unsupported bounds type")
    #
    #     # 处理不同参考系
    #     if rng.referential == OffsetReferential.ORIGINAL:
    #         # 获取原始字符串的总字节数（预计算存储）
    #         total_original_bytes = self.original_byte_length
    #         # 解析原始范围边界
    #         start, stop = _resolve_bounds(rng.bounds, total_original_bytes)
    #         # 收集所有规范化字节索引（其原始范围完全在目标范围内）
    #         matching_indices = [
    #             i for i, (a_start, a_end) in enumerate(self.alignments)
    #             if a_start >= start and a_end <= stop
    #         ]
    #         # 检查索引连续性
    #         if not matching_indices:
    #             return None
    #         for i in range(1, len(matching_indices)):
    #             if matching_indices[i] != matching_indices[i - 1] + 1:
    #                 return None
    #         return range(matching_indices[0], matching_indices[-1] + 1)
    #
    #     elif rng.referential == OffsetReferential.NORMALIZED:
    #         # 获取规范化字符串的总字节数（即alignments长度）
    #         total_norm_bytes = len(self.alignments)
    #         # 解析规范化范围边界
    #         start, stop = _resolve_bounds(rng.bounds, total_norm_bytes)
    #         # 检查有效性
    #         if start >= total_norm_bytes or stop > total_norm_bytes or start > stop:
    #             return None
    #         return range(start, stop)
    #
    #     else:
    #         raise ValueError("Invalid referential type")
    #
    # def _get_original_start(self, norm_byte: int) -> int:
    #     """根据规范化字节位置计算原始字符串的起始字节"""
    #     return self.alignments[norm_byte][0] if self.alignments else 0
    # endregion


    # region Helper Methods

    def __len__(self) -> int:
        return len(self.normalized)

    def len_original(self) -> int:
        return len(self.original)
    # endregion


# region Utility Functions
def get_range_of(s: str, rng: range) -> str:
    chars = list(s)
    return ''.join(chars[rng.start:rng.stop])


def bytes_to_char(s: str, byte_range: range) -> Optional[range]:
    char_pos = []
    current_byte = 0
    for i, c in enumerate(s):
        char_len = len(c.encode('utf-8'))
        if current_byte >= byte_range.start and current_byte + char_len <= byte_range.stop:
            char_pos.append(i)
        current_byte += char_len
    if not char_pos:
        return None
    return range(char_pos[0], char_pos[-1] + 1)


def char_to_bytes(s: str, char_range: range) -> Optional[range]:
    byte_start = 0
    byte_end = 0
    for i, c in enumerate(s):
        char_len = len(c.encode('utf-8'))
        if i < char_range.start:
            byte_start += char_len
        if i < char_range.stop:
            byte_end += char_len
    return range(byte_start, byte_end)
# endregion

if __name__ == '__main__':
    # region unicode test
    print("+----------------- nfd ----------------+")
    ns = NormalizedString("café")   # 原始5字节
    print(f"str: {ns.get()}, alignment: {ns.alignments}")
    ns.nfd()    # 规范化后6字节
    print(f"str: {ns.get()}, alignment: {ns.alignments}")   # [(0, 1), (1, 2), (2, 3), (3, 5), (3, 5), (3, 5)]

    print("+----------------- nfc ----------------+")
    ns = NormalizedString("cafe\u0301")
    print(f"str: {ns.get()}, alignment: {ns.alignments}")
    ns.nfc()
    print(f"str: {ns.get()}, alignment: {ns.alignments}")   # [(0, 1), (1, 2), (2, 3), (3, 6), (3, 6)]

    print("+----------------- nfkc ----------------+")
    ns = NormalizedString("①")  # 原始3字节： \xef\xbc\x91
    print(f"str: {ns.get()}, alignment: {ns.alignments}")
    ns.nfkc()   # 规范化后1字节：
    print(f"str: {ns.get()}, alignment: {ns.alignments}")   # 输出：1   [(0, 3)]

    print("+----------------- nfkd ----------------+")
    ns = NormalizedString("㍍")  # 原始3字节： \xe3\x8d\x8d
    print(f"str: {ns.get()}, alignment: {ns.alignments}")
    ns.nfkd()   # 规范化后12字节：
    print(f"str: {ns.get()}, alignment: {ns.alignments}")   # 输出：メートル   [(0, 3)] * 12
    # endregion

    # region
    print("+----------------- slice ----------------+")
    ns = NormalizedString("café")
    sliced = ns.slice(Range.Normalized(slice(3, 5)))    # 提取3-5字节，即'e\u0301'
    print(f"str: {sliced.get()}, alignment: {sliced.alignments}")   # 输出：e\u0301   [(3, 5), (3, 5)]

    print("+----------------- 跨字符slice ----------------+")
    ns = NormalizedString("café")
    sliced = ns.slice(Range.Normalized(range(3, 4)))  # 试图切分é的中间字节
    print(f"sliced is {sliced}")  # 输出：None

    print("+----------------- filter ----------------+")
    ns = NormalizedString("a1b2c3")
    filterd = ns.filter(lambda x: not x.isdigit())  # 过滤数字
    print(f"str: {filterd.get()}, alignment: {filterd.alignments}")  # 输出：abc    [(0, 1), (2, 3), (4, 5)]

    print("+----------------- 多字节的filter ----------------+")
    ns = NormalizedString("café")
    filterd = ns.filter(lambda x: x != 'é')  # 删除é
    print(f"str: {filterd.get()}, alignment: {filterd.alignments}")  # 输出：caf   [(0, 1), (1, 2), (2, 3)]

    print("+----------------- 多字节的filter与strip ----------------+")
    import unicodedata
    ns = NormalizedString("  Héllo  ")
    ns.nfd().filter(lambda c: not unicodedata.combining(c))
    print(f"str: {ns.get()}, alignment: {ns.alignments}")
    ns.strip()
    print(f"str: {ns.get()}, alignment: {ns.alignments}") # 输出："Hello"

    print("+----------------- map ----------------+")
    ns = NormalizedString("hello")
    mapped = ns.map(lambda x: x.upper())  # 全部变成大写
    print(f"str: {mapped.get()}, alignment: {mapped.alignments}")  # 输出：HELLO

    print("+----------------- 多字符map ----------------+")
    s = NormalizedString("a→b")
    s.map(lambda c: "→" if c == "a" else "←")  # a替换为→，其他替换为←
    print(f"str: {s.get()}, alignment: {s.alignments}")  # 输出 "→←←"    [(0, 1), (0, 1), (0, 1), (1, 4), (1, 4), (1, 4), (4, 5), (4, 5), (4, 5)]

    print("+----------------- 大小写转换 ----------------+")
    ns = NormalizedString("straB").lowercase()
    print(f"str: {ns.get()}, alignment: {ns.alignments}")

    pass
