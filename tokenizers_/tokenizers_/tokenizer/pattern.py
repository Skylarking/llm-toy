import re
from typing import List, Tuple, Union, Callable, Optional
from dataclasses import dataclass
from mod import Offsets

SysRegex = re.Pattern  # 系统正则表达式类型

def char_to_byte_offsets(s: str) -> List[int]:
    """生成字符索引到字节偏移的映射表"""
    byte_offsets = [0]
    for c in s:
        byte_offsets.append(byte_offsets[-1] + len(c.encode('utf-8')))
    return byte_offsets

class Pattern:
    """支持多字节字符的模式匹配基类"""

    def find_matches(self, inside: str) -> List[Tuple[Offsets, bool]]:
        raise NotImplementedError


@dataclass
class Invert(Pattern):
    """反转匹配结果的包装类"""
    pattern: Pattern

    def find_matches(self, inside: str) -> List[Tuple[Offsets, bool]]:
        matches = self.pattern.find_matches(inside)
        return [(offsets, not flag) for (offsets, flag) in matches]


class CharPattern(Pattern):
    """修复后的字符匹配模式，正确处理多字节字符"""

    def __init__(self, char: str):
        assert len(char) == 1, "Char pattern must be single character"
        self.char = char
        # 预计算目标字符的字节长度
        self.char_bytes = char.encode('utf-8')
        self.byte_len = len(self.char_bytes)

    def find_matches(self, inside: str) -> List[Tuple[Offsets, bool]]:
        if not inside:
            return [((0, 0), False)]

        # 生成字符索引到字节偏移的映射表
        byte_offsets = char_to_byte_offsets(inside)
        matches = []
        prev_byte = 0

        # 遍历每个字符并生成独立区间
        for char_idx in range(len(inside)):
            c = inside[char_idx]
            start_byte = byte_offsets[char_idx]
            end_byte = byte_offsets[char_idx + 1]

            # 前一个区间的结束位置与当前开始位置不连续时，填充间隔
            if prev_byte < start_byte:
                matches.append(((prev_byte, start_byte), False))

            # 当前字符匹配结果
            is_match = (c == self.char)
            matches.append(((start_byte, end_byte), is_match))
            prev_byte = end_byte

        # 处理最后一个字符之后的剩余部分
        if prev_byte < byte_offsets[-1]:
            matches.append(((prev_byte, byte_offsets[-1]), False))

        return matches

class StrPattern(Pattern):
    """支持多字节的字符串字面量匹配模式"""

    def __init__(self, pattern: str):
        self.pattern = pattern
        self.byte_pattern = pattern.encode('utf-8')

    def find_matches(self, inside: str) -> List[Tuple[Offsets, bool]]:
        if not self.pattern:
            return [((0, len(inside.encode('utf-8'))), False)]

        byte_str = inside.encode('utf-8')
        matches = []
        prev = 0
        start = 0

        # 使用字节级搜索确保准确匹配
        while start <= len(byte_str):
            pos = byte_str.find(self.byte_pattern, start)
            if pos == -1:
                break
            end = pos + len(self.byte_pattern)

            if prev < pos:
                matches.append(((prev, pos), False))
            matches.append(((pos, end), True))

            prev = end
            start = end

        if prev < len(byte_str):
            matches.append(((prev, len(byte_str)), False))

        return matches


class RegexPattern(Pattern):
    """支持多字节字符的正则匹配模式"""

    def __init__(self, pattern: Union[str, re.Pattern]):
        if isinstance(pattern, str):
            # 确保正则表达式使用Unicode模式
            self.regex = re.compile(pattern, flags=re.UNICODE)
        else:
            self.regex = pattern

    def find_matches(self, inside: str) -> List[Tuple[Offsets, bool]]:
        if not inside:
            return [((0, 0), False)]

        byte_offsets = char_to_byte_offsets(inside) # 获取每个字符的字节偏移量
        matches = []
        prev_byte = 0

        # 在字符级进行匹配，转换为字节偏移
        for match in self.regex.finditer(inside):
            start_char = match.start()
            end_char = match.end()

            start_byte = byte_offsets[start_char]
            end_byte = byte_offsets[end_char]

            if prev_byte < start_byte:
                matches.append(((prev_byte, start_byte), False))
            matches.append(((start_byte, end_byte), True))
            prev_byte = end_byte

        if prev_byte < byte_offsets[-1]:
            matches.append(((prev_byte, byte_offsets[-1]), False))

        return matches


class FunctionPattern(Pattern):
    """支持多字节字符的函数式匹配"""

    def __init__(self, func: Callable[[str], bool]):
        self.func = func

    def find_matches(self, inside: str) -> List[Tuple[Offsets, bool]]:
        if not inside:
            return [((0, 0), False)]

        byte_offsets = char_to_byte_offsets(inside)
        matches = []
        prev_byte = 0

        for char_idx, c in enumerate(inside):
            if self.func(c):
                start_byte = byte_offsets[char_idx]
                end_byte = byte_offsets[char_idx + 1]

                if prev_byte < start_byte:
                    matches.append(((prev_byte, start_byte), False))
                matches.append(((start_byte, end_byte), True))
                prev_byte = end_byte

        if prev_byte < byte_offsets[-1]:
            matches.append(((prev_byte, byte_offsets[-1]), False))

        return matches


def pattern(p: Union[str, re.Pattern, Callable]) -> Pattern:
    """工厂函数（自动处理多字节）"""
    if isinstance(p, str):
        if len(p) == 1:
            return CharPattern(p)
        else:
            return StrPattern(p)
    elif isinstance(p, re.Pattern):
        return RegexPattern(p)
    elif callable(p):
        return FunctionPattern(p)
    raise ValueError(f"Unsupported pattern type: {type(p)}")


# 测试用例
if __name__ == "__main__":
    def test_chinese():
        # 测试中文字符
        text = "中文测试"

        # 测试单字符匹配
        p = pattern("文")
        assert p.find_matches(text) == [
            ((0, 3), False),  # "中" 的字节范围 0-3
            ((3, 6), True),  # "文" 匹配成功
            ((6, 9), False),  # "测"
            ((9, 12), False)  # "试"
        ]

        # 测试正则表达式
        p = pattern(re.compile(r"\w+"))
        assert p.find_matches(text) == [
            ((0, 12), True)  # 整个字符串匹配\w+
        ]

        # 测试函数匹配
        p = pattern(lambda c: c == "测")
        assert p.find_matches(text) == [
            ((0, 6), False),  # "中文"
            ((6, 9), True),  # "测"
            ((9, 12), False)  # "试"
        ]


    test_chinese()
    print("所有多字节测试通过!")
