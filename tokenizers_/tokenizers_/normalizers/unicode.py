import unicodedata
from typing import List, Tuple
from tokenizers_.tokenizers.normalizers import NormalizedString


class Normalizer:
    def normalize(self, normalized_str: NormalizedString) -> None:
        raise NotImplementedError


# Unicode 规范化实现 --------------------------------------------------------
class NFD(Normalizer):
    def normalize(self, nstr: NormalizedString) -> None:
        nstr.nfd()


class NFKD(Normalizer):
    def normalize(self, nstr: NormalizedString) -> None:
        nstr.nfkd()


class NFC(Normalizer):
    def normalize(self, nstr: NormalizedString) -> None:
        nstr.nfc()


class NFKC(Normalizer):
    def normalize(self, nstr: NormalizedString) -> None:
        nstr.nfkc()


# NMT 规范化实现 ----------------------------------------------------------
def do_nmt(text: str) -> str:
    """处理控制字符和特殊空格"""
    filtered = []
    for c in text:
        code = ord(c)
        # 过滤控制字符
        if code in {
            0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007, 0x0008,
            0x000B, 0x000E, 0x000F, 0x0010, 0x0011, 0x0012, 0x0013, 0x0014,
            0x0015, 0x0016, 0x0017, 0x0018, 0x0019, 0x001A, 0x001B, 0x001C,
            0x001D, 0x001E, 0x001F, 0x007F, 0x008F, 0x009F
        }:
            continue

        # 替换特殊空格
        replaced = c
        if code in {
            0x0009, 0x000A, 0x000C, 0x000D,
            0x1680, 0x200B, 0x200C, 0x200D, 0x200E, 0x200F,
            0x2028, 0x2029, 0x2581, 0xFEFF, 0xFFFD
        }:
            replaced = ' '

        filtered.append(replaced)

    return ''.join(filtered)


class Nmt(Normalizer):
    def normalize(self, nstr: NormalizedString) -> None:
        nstr.normalized = do_nmt(nstr.normalized)


# 测试用例 ----------------------------------------------------------------
if __name__ == "__main__":
    # 正确写法：使用Python的标准Unicode转义格式 \uXXXX
    nstr = NormalizedString("\ufb01")  # 直接使用 \ufb01 表示 ﬁ 字符
    NFKC().normalize(nstr)
    print(f"NFKC: {nstr.normalized}")  # 输出: fi

    # 正确表示包含控制字符的字符串
    text = "Hello\u0009world\u0001"    # \u0009=制表符，\u0001=控制字符
    nstr = NormalizedString(text)
    Nmt().normalize(nstr)
    print(f"Nmt: {nstr.normalized!r}") # 输出: 'Hello world '

