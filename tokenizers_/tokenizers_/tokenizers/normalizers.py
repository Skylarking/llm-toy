import unicodedata
from typing import List, Tuple


class NormalizedString:
    def __init__(self, original: str):
        self.original = original
        self.normalized = original
        # 对齐信息：每个字符对应原始字符串的 (start, end) 索引
        self.alignments: List[Tuple[int, int]] = [(i, i + 1) for i in range(len(original))]
        self.original_shift = 0  # 原始字符串的偏移量

    def get(self) -> str:
        return self.normalized

    def _normalize(self, form: str) -> 'NormalizedString':
        """内部规范化方法"""
        original_normalized = self.normalized
        new_normalized = unicodedata.normalize(form, original_normalized)

        # 生成转换操作列表
        transformations = []
        src_idx = 0

        # 逐个字符比较生成变换指令
        for dst_char in new_normalized:
            if src_idx < len(original_normalized) and dst_char == original_normalized[src_idx]:
                # 保留原字符
                transformations.append((dst_char, 0))
                src_idx += 1
            else:
                # 插入新字符
                transformations.append((dst_char, 1))

        # 处理可能的删除操作（当新字符串比原字符串短时）
        delete_count = len(original_normalized) - src_idx
        if delete_count > 0:
            # 添加删除指令（Python实现简化处理）
            transformations.append(('', -delete_count))

        self._apply_transformations(transformations)
        return self

    def _apply_transformations(self, transformations: List[Tuple[str, int]]):
        """应用转换操作更新对齐信息"""
        new_normalized = []
        new_alignments = []
        current = 0  # 当前处理位置

        for char, change in transformations:
            if change == 0:  # 保留字符
                if current < len(self.alignments):
                    align = self.alignments[current]
                    new_normalized.append(char)
                    new_alignments.append(align)
                    current += 1
            elif change > 0:  # 插入字符
                # 继承前一个字符的对齐信息
                prev_align = self.alignments[current - 1] if current > 0 else (0, 0)
                new_normalized.append(char)
                new_alignments.append(prev_align)
            elif change < 0:  # 删除字符
                current += abs(change)

        self.normalized = ''.join(new_normalized)
        self.alignments = new_alignments

    def nfd(self) -> 'NormalizedString':
        """应用 NFD 规范化 (规范分解)"""
        return self._normalize('NFD')

    def nfkd(self) -> 'NormalizedString':
        """应用 NFKD 规范化 (兼容分解)"""
        return self._normalize('NFKD')

    def nfc(self) -> 'NormalizedString':
        """应用 NFC 规范化 (规范组合)"""
        return self._normalize('NFC')

    def nfkc(self) -> 'NormalizedString':
        """应用 NFKC 规范化 (兼容组合)"""
        return self._normalize('NFKC')

    def __repr__(self):
        return f"NormalizedString(original={self.original!r}, normalized={self.normalized!r})"


# 测试用例
if __name__ == "__main__":
    # NFD 测试：分解字符
    nstr = NormalizedString("é")  # U+00E9
    nstr.nfd()
    print("NFD 结果:", nstr.normalized)  # 应输出 'e\u0301'

    # NFC 测试：组合字符
    decomposed = NormalizedString("e\u0301")  # e + combining acute
    decomposed.nfc()
    print("NFC 结果:", decomposed.normalized)  # 应输出 'é' (U+00E9)

    # NFKD 测试：兼容分解
    ligature = NormalizedString("ﬃ")  # U+FB03
    ligature.nfkd()
    print("NFKD 结果:", ligature.normalized)  # 应输出 'ffi'

    # NFKC 测试：兼容组合
    decomposed = NormalizedString("²")  # 上标 2
    decomposed.nfkc()
    print("NFKC 结果:", decomposed.normalized)  # 应输出 '2'
