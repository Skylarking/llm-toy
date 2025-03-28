from sympy.external.gmpy import invert

from base import PreTokenizer
from tokenizers_.tokenizer import PreTokenizedString, Split, NormalizedString, SplitDelimiterBehavior, OffsetReferential, OffsetType, Range, pattern, Invert
import re
from typing import List, Tuple, Iterable

class Whitespace:
    """基于正则表达式\\w+|[^\\w\\s]+分割文本（兼容Rust版本行为）"""

    def __init__(self):
        # 编译正则表达式（匹配单词字符或非单词非空白字符）
        self.pattern = pattern(re.compile(r'(\w+)|([^\w\s]+)', flags=re.ASCII)) # 注意：rust和python的\w含义不一样，在rust中\w等价于[A-Za-z0-9_]

    def pre_tokenize(self, pretokenized: PreTokenizedString) -> None:
        """核心分割逻辑"""

        def split_func(_: int, normalized: NormalizedString) -> Iterable[Split]:
            # 执行分割（REMOVED行为，即去除空白符
            parts = normalized.split(Invert(self.pattern), SplitDelimiterBehavior.REMOVED)

            # 生成Split对象并过滤空结果
            return [
                Split.from_normalized(part)
                for part in parts
                if not part.is_empty()
            ]

        pretokenized.split(split_func)

    def pre_tokenize_str(self, sequence: str) -> List[Tuple[str, Tuple[int, int]]]:
        """快速预分词接口"""
        pretokenized = PreTokenizedString(sequence)
        self.pre_tokenize(pretokenized)

        # 提取结果（使用ORIGINAL参考系和BYTE偏移类型）
        splits = pretokenized.get_splits(
            OffsetReferential.ORIGINAL,
            OffsetType.BYTE
        )

        # 转换格式为 [(text, (start, end)), ...] 并过滤空结果
        return [
            (text, offsets)
            for text, offsets, _ in splits
            if text  # 过滤空字符串
        ]

class WhitespaceSplit(PreTokenizer):
    """基于空白字符分割文本"""

    def pre_tokenize(self, pretokenized: PreTokenizedString) -> None:
        """执行预分词操作"""

        def split_func(_: int, normalized: NormalizedString) -> Iterable[Split]:
            parts = []
            # 获取规范化字符串
            s = normalized.get()
            # 找到所有空白分隔符位置
            splits = [i for i, c in enumerate(s) if c.isspace()]
            splits.append(len(s))  # 添加末尾哨兵

            prev = 0
            for split in splits:
                if prev < split:
                    parts.append(Split.from_normalized(normalized.slice(prev, split)))
                prev = split + 1  # 跳过空白符

            return parts

        pretokenized.split(split_func)

    def pre_tokenize_str(self, sequence: str) -> List[Tuple[str, Tuple[int, int]]]:
        """复用PreTokenizedString处理流程"""
        pretokenized = PreTokenizedString(sequence)
        self.pre_tokenize(pretokenized)

        # 获取字符级偏移（演示不同偏移类型）
        splits = pretokenized.get_splits(
            OffsetReferential.NORMALIZED,
            OffsetType.CHAR
        )

        return [
            (text, offsets)
            for text, offsets, _ in splits
        ]


# 测试用例
if __name__ == "__main__":
    pretok = Whitespace()
    assert pretok.pre_tokenize_str("Hey man!") == [
        ("Hey", (0, 3)), ("man", (4, 7)), ("!", (7, 8))
    ]

    # 测试WhitespaceSplit版本
    # split_pretok = WhitespaceSplit()
    # assert split_pretok.pre_tokenize_str("Hey man!") == [
    #     ("Hey", (0, 3)), ("man!", (4, 8))
    # ]

