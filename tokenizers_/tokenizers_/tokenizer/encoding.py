from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Iterable, Iterator
import copy
from .mod import Token, Offsets

# 类型别名
PaddingDirection = str  # "left" 或 "right"
TruncationDirection = str  # "left" 或 "right"


@dataclass
class Encoding:
    """表示分词器的输出结果"""
    ids: List[int] = field(default_factory=list)    # 存储每一个token的id
    type_ids: List[int] = field(default_factory=list)   # 区分不同句子或者文本段的标识符（例如BERT模型），例如【0, 0, 0, 0, 1, 1, 1]表示前4个是一个句子，后三个是另一个句子
    tokens: List[str] = field(default_factory=list) # 原始的token，和ids长度一样，是一一对应的
    words: List[Optional[int]] = field(default_factory=list)    # token到vocab的索引映射，其中None表示一个特殊token
    offsets: List[Offsets] = field(default_factory=list)    # 记录每个token在原始文本中的偏移量，[(0, 5), (6, 11)]表示第一个token占0～5字符位置
    special_tokens_mask: List[int] = field(default_factory=list)    # 标识哪些位置是特殊Token，例如[1, 0, 0, 1]其中1表示该位置是特殊token
    attention_mask: List[int] = field(default_factory=list) # 表示哪些token需要被模型关注，例如[1, 1, 1, 0, 0]其中后两个0表示padding
    overflowing: List[Encoding] = field(default_factory=list)   # 存储因截断产生的溢出编码，当文本被截断为512个token时，保存后续溢出的部分
    sequence_ranges: Dict[int, range] = field(default_factory=dict) # 记录不同序列在编码中的位置范围，例如{0: range(0, 10), 1: range(10, 20)}表示两个序列的位置。处理多序列输入时用到，如问答中的问题+上下文

    # region 构造函数
    @classmethod
    def new(
            cls,
            ids: List[int],
            type_ids: List[int],
            tokens: List[str],
            words: List[Optional[int]],
            offsets: List[Offsets],
            special_tokens_mask: List[int],
            attention_mask: List[int],
            overflowing: List[Encoding],
            sequence_ranges: Dict[int, range]
    ) -> Encoding:
        return cls(ids, type_ids, tokens, words, offsets,
                   special_tokens_mask, attention_mask, overflowing, sequence_ranges)

    @classmethod
    def with_capacity(cls, capacity: int) -> Encoding:
        return cls(
            ids=[],
            type_ids=[],
            tokens=[],
            words=[],
            offsets=[],
            special_tokens_mask=[],
            attention_mask=[],
            overflowing=[],
            sequence_ranges={}
        )

    @classmethod
    def from_tokens(cls, tokens: List[Token], type_id: int) -> 'Encoding':
        """从Token列表构造Encoding对象"""
        return cls(
            ids=[t.id for t in tokens],
            type_ids=[type_id] * len(tokens),
            tokens=[t.value for t in tokens],
            words=[None] * len(tokens),
            offsets=[t.offsets for t in tokens],
            special_tokens_mask=[0] * len(tokens),
            attention_mask=[1] * len(tokens)
        )

    # endregion

    # region 属性方法
    @property
    def is_empty(self) -> bool:
        return len(self.ids) == 0

    @property
    def len(self) -> int:
        return len(self.ids)

    @property
    def n_sequences(self) -> int:
        return len(self.sequence_ranges) if self.sequence_ranges else 1

    def sequence_range(self, sequence_id: int) -> range:
        return self.sequence_ranges.get(sequence_id, range(0, self.len))

    # endregion

    # region 操作方法
    def set_sequence_id(self, sequence_id: int) -> None:
        self.sequence_ranges[sequence_id] = range(0, self.len())

    def truncate(self, max_len: int, stride: int, direction: TruncationDirection) -> None:
        if max_len >= self.len:
            return

        if max_len == 0:
            self.overflowing.append(copy.deepcopy(self))
            self.__dict__.update(Encoding.with_capacity(0).__dict__)
            return

        # 实现截断逻辑（简化为右侧截断）
        keep = slice(0, max_len) if direction == "right" else slice(-max_len, None)
        overflow = self._slice(keep, stride)

        self.overflowing.append(overflow)
        self._apply_slice(keep)

    def _slice(self, slice_range: slice, stride: int) -> Encoding:
        """生成截断后的溢出部分"""
        return Encoding(
            ids=self.ids[slice_range],
            type_ids=self.type_ids[slice_range],
            tokens=self.tokens[slice_range],
            words=self.words[slice_range],
            offsets=self.offsets[slice_range],
            special_tokens_mask=self.special_tokens_mask[slice_range],
            attention_mask=self.attention_mask[slice_range]
        )

    def _apply_slice(self, slice_range: slice) -> None:
        """应用切片到当前对象"""
        self.ids = self.ids[slice_range]
        self.type_ids = self.type_ids[slice_range]
        self.tokens = self.tokens[slice_range]
        self.words = self.words[slice_range]
        self.offsets = self.offsets[slice_range]
        self.special_tokens_mask = self.special_tokens_mask[slice_range]
        self.attention_mask = self.attention_mask[slice_range]

    def pad(self, target_length: int, pad_id: int, pad_type_id: int,
            pad_token: str, direction: PaddingDirection) -> None:
        pad_len = target_length - self.len
        if pad_len <= 0:
            return

        pad_data = {
            "ids": [pad_id] * pad_len,
            "type_ids": [pad_type_id] * pad_len,
            "tokens": [pad_token] * pad_len,
            "words": [None] * pad_len,
            "offsets": [(0, 0)] * pad_len,
            "special_tokens_mask": [1] * pad_len,
            "attention_mask": [0] * pad_len
        }

        if direction == "left":
            for field, values in pad_data.items():
                getattr(self, field)[:0] = values
        else:
            for field, values in pad_data.items():
                getattr(self, field).extend(values)

    # endregion

    # region 迭代器支持
    @classmethod
    def merge(cls, encodings: Iterable[Encoding], growing_offsets: bool) -> Encoding:
        merged = Encoding()
        for enc in encodings:
            merged.merge_with(enc, growing_offsets)
        return merged

    def merge_with(self, other: Encoding, growing_offsets: bool) -> None:
        self.ids.extend(other.ids)
        self.type_ids.extend(other.type_ids)
        self.tokens.extend(other.tokens)
        self.words.extend(other.words)
        self.special_tokens_mask.extend(other.special_tokens_mask)
        self.attention_mask.extend(other.attention_mask)
        self.overflowing.extend(other.overflowing)

        # 处理 offsets
        base = self.offsets[-1][1] if growing_offsets and self.offsets else 0
        self.offsets.extend([(s + base, e + base) for s, e in other.offsets])

    def __iter__(self) -> Iterator[tuple]:
        return zip(self.ids, self.tokens, self.offsets, self.words, self.type_ids)
    # endregion

    def word_to_tokens(self, word: int, sequence_id: int) -> Optional[Tuple[int, int]]:
        """获取指定单词对应的token范围 (start, end)"""
        seq_range = self.sequence_range(sequence_id)
        if seq_range is None:
            return None

        start_token = None
        end_token = None

        # 遍历序列范围内的token
        for rel_pos, word_id in enumerate(self.words[seq_range.start: seq_range.stop]):
            abs_pos = seq_range.start + rel_pos
            if word_id == word:
                if start_token is None or rel_pos < start_token:
                    start_token = rel_pos
                end_token = rel_pos + 1  # 结束位置是exclusive的

        if start_token is None or end_token is None:
            return None

        return (
            seq_range.start + start_token,
            seq_range.start + end_token
        )

    def word_to_chars(self, word: int, sequence_id: int) -> Optional[Offsets]:
        """获取单词的字符偏移量"""
        tokens_range = self.word_to_tokens(word, sequence_id)
        if tokens_range is None:
            return None

        start_token, end_token = tokens_range
        if end_token == 0:
            return None

        return (
            self.offsets[start_token][0],
            self.offsets[end_token - 1][1]
        )

    def token_to_sequence(self, token_idx: int) -> Optional[int]:
        """查找token所属的序列ID"""
        if token_idx >= self.len:
            return None

        if not self.sequence_ranges:
            return 0

        for seq_id, seq_range in self.sequence_ranges.items():
            if seq_range.start <= token_idx < seq_range.stop:
                return seq_id
        return None

    def token_to_word(self, token_idx: int) -> Optional[Tuple[int, int]]:
        """获取token对应的（序列ID, 单词ID）"""
        seq_id = self.token_to_sequence(token_idx)
        if seq_id is None:
            return None

        word_id = self.words[token_idx]
        if word_id is None:
            return None

        return (seq_id, word_id)

    def token_to_chars(self, token_idx: int) -> Optional[Tuple[int, Offsets]]:
        """获取token的序列ID和字符偏移量"""
        seq_id = self.token_to_sequence(token_idx)
        if seq_id is None or token_idx >= len(self.offsets):
            return None

        return (seq_id, self.offsets[token_idx])

    def char_to_token(self, char_pos: int, sequence_id: int) -> Optional[int]:
        """根据字符位置查找对应token"""
        seq_range = self.sequence_range(sequence_id)
        if seq_range is None:
            return None

        for rel_pos, (start, end) in enumerate(self.offsets[seq_range.start: seq_range.stop]):
            if start <= char_pos < end:
                return seq_range.start + rel_pos
        return None

    def char_to_word(self, char_pos: int, sequence_id: int) -> Optional[int]:
        """根据字符位置查找对应单词ID"""
        token_idx = self.char_to_token(char_pos, sequence_id)
        if token_idx is None:
            return None

        word_info = self.token_to_word(token_idx)
        return word_info[1] if word_info else None

    # 辅助方法
    # def sequence_range(self, sequence_id: int) -> Optional[range]:
    #     """获取指定序列的范围"""
    #     if not self.sequence_ranges:
    #         return range(0, self.len()) if sequence_id == 0 else None
    #     return self.sequence_ranges.get(sequence_id)
    #
    # def len(self) -> int:
    #     """返回编码长度"""
    #     return len(self.ids)


if __name__ == '__main__':
    # 截断示例
    enc = Encoding(ids=[1, 2, 3, 4, 5])
    enc.truncate(3, 1, "right")
    print(enc.ids)  # [1,2,3]
    print(enc.overflowing)  # 包含[2,3,4,5]的Encoding

    # 填充示例
    enc.pad(5, 0, 0, "[PAD]", "right")
    print(enc.ids)  # [1,2,3,0,0]

    # -------------------- 典型的工作流程-----------------------
    print(f"+-------------------- 典型的工作流程-----------------------+")
    # 初始化编码
    enc = Encoding()

    # 添加第一个句子
    enc.ids.extend([101, 2342, 1037, 102])  # [CLS] hello world [SEP]
    enc.type_ids.extend([0] * 4)
    enc.tokens.extend(["[CLS]", "hello", "world", "[SEP]"])
    enc.words.extend([None, 0, 0, None])  # "hello world" 被分成两个token
    enc.offsets.extend([(0, 0), (0, 5), (6, 11), (0, 0)])
    enc.special_tokens_mask.extend([1, 0, 0, 1])
    enc.attention_mask.extend([1] * 4)

    # 设置序列范围
    enc.sequence_ranges[0] = range(0, 4)

    # 处理溢出
    overflow = Encoding.from_tokens([Token(1, 'hello', (0, 5))], 1)
    enc.overflowing.append(overflow)

    """
    每个字段的设计都服务于以下核心需求：
    完整保留编码信息：存储从原始文本到模型输入的全链路数据
    支持复杂处理：截断、填充、多序列合并等操作需要元数据支持
    便于后处理：对齐预测结果与原始文本需要字符 / 单词级定位信息
    高效序列化：结构化存储便于持久化和传输
    特殊字段的协同工作示例（定位单词"world"）：
    """
    # 通过words找到token范围
    word_id = 0  # 假设要查找第一个单词
    token_start, token_end = enc.word_to_tokens(word_id, sequence_id=0)

    # 通过offsets获取字符位置
    char_start = enc.offsets[token_start][0]
    char_end = enc.offsets[token_end - 1][1]

    # 通过sequence_ranges验证是否跨序列
    for seq_id, seq_range in enc.sequence_ranges.items():
        if token_start in seq_range:
            print(f"Word located in sequence {seq_id}")

    # -------------------- word char token映射方法测试 -----------------------
    print(f"+-------------------- word char token映射方法测试 -----------------------+")


    def test_encoding_mappings():
        # 创建测试编码
        enc = Encoding(
            ids=[1, 2, 3, 4, 5],
            type_ids=[0, 0, 0, 0, 0],
            tokens=["He", "llo", "world", "!", "How"],
            words=[0, 0, 1, 2, 3],  # 单词索引
            offsets=[(0, 2), (2, 5), (6, 11), (11, 12), (0, 3)],
            sequence_ranges={
                0: range(0, 4),  # 第一个序列包含前4个token
                1: range(4, 5)  # 第二个序列包含最后一个token
            }
        )

        # 测试单词到token的映射
        assert enc.word_to_tokens(0, 0) == (0, 2)  # "He"+"llo"对应单词0
        assert enc.word_to_tokens(1, 0) == (2, 3)  # "world"对应单词1
        assert enc.word_to_tokens(3, 1) == (4, 5)  # 第二个序列的单词3

        # 测试字符到token的映射
        assert enc.char_to_token(3, 0) == 1  # 位置3在"llo"中
        assert enc.char_to_token(7, 0) == 2  # 位置7在"world"中
        assert enc.char_to_token(2, 1) == 4  # 第二个序列的"How"

        # 测试token到单词的反向映射
        assert enc.token_to_word(1) == (0, 0)  # 第1个token属于序列0的单词0
        assert enc.token_to_word(4) == (1, 3)  # 第4个token属于序列1的单词3

        # 测试边界情况
        assert enc.word_to_tokens(5, 0) is None  # 不存在的单词
        assert enc.char_to_token(20, 0) is None  # 超出范围的字符位置
        print("done!")

    test_encoding_mappings()

