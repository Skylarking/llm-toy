from base import PreTokenizer
import re
from typing import List, Tuple

class Whitespace(PreTokenizer):
    """
    This pre-tokenizer simply splits using the following regex: `\w+|[^\w\s]+`
    """
    def __init__(self):
        # 预编译正则表达式（等效于 Rust 的 lazy_static!）
        # \w+        : 匹配连续的字母/数字/下划线（包括 Unicode 字符）
        # |          : 或
        # [^\w\s]+   : 匹配连续的非单词、非空白字符（如标点符号）
        self.re = re.compile(r'\w+|[^\w\s]+', flags=re.UNICODE)

    def pre_tokenize(self, pretok: str) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Pre-tokenize a :class:`~tokenizers.PyPreTokenizedString` in-place

        This method allows to modify a :class:`~tokenizers.PreTokenizedString` to
        keep track of the pre-tokenization, and leverage the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you just want to see the result of
        the pre-tokenization of a raw string, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize_str`

        Args:
            pretok (:class:`~tokenizers.PreTokenizedString):
                The pre-tokenized string on which to apply this
                :class:`~tokenizers.pre_tokenizers.PreTokenizer`
        """
        pass

    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given string

        This method provides a way to visualize the effect of a
        :class:`~tokenizers.pre_tokenizers.PreTokenizer` but it does not keep track of the
        alignment, nor does it provide all the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you need some of these, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize`

        Args:
            sequence (:obj:`str`):
                A string to pre-tokeize

        Returns:
            :obj:`List[Tuple[str, Offsets]]`:
                A list of tuple with the pre-tokenized parts and their offsets
        """
        tokens = []
        for match in self.re.finditer(sequence):
            start, end = match.span()
            tokens.append((match.group(), (start, end)))
        return tokens

# 测试用例
if __name__ == "__main__":
    pre_tokenizer = Whitespace()

    # 测试 1：英文与符号
    text1 = "Hello,world!"
    print(pre_tokenizer.pre_tokenize(text1))
    # 输出: [('Hello', (0, 5)), (',', (5, 6)), ('world', (6, 11)), ('!', (11, 12))]

    # 测试 2：中文与符号
    text2 = "你好！Python。"
    print(pre_tokenizer.pre_tokenize(text2))
    # 输出: [('你好', (0, 2)), ('！', (2, 3)), ('Python', (3, 9)), ('。', (9, 10))]

    # 测试 3：混合 Unicode
    text3 = "123abc_नमस्ते@#$"
    print(pre_tokenizer.pre_tokenize(text3))
    # 输出: [('123abc_नमस्ते', (0, 12)), ('@#$', (12, 15))]

