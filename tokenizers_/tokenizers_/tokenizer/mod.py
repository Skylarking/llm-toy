from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Iterable, Iterator

Offsets = Tuple[int, int]

@dataclass(frozen=True, order=True)
class Token:
    """不可变数据类"""
    id: int
    value: str
    offsets: tuple[int, int]

    @classmethod
    def new(cls, id: int, value: str, offsets: tuple[int, int]) -> 'Token':
        """工厂方法, 构造函数"""
        return cls(id, value, offsets)

    def __str__(self) -> str:
        """字符串表示"""
        return f"Token(id={self.id}, value='{self.value}', offsets={self.offsets})"

if __name__ == '__main__':
    t1 = Token(1, 'hello', (0, 5))

    t2 = Token.new(2, 'world', (6, 11))

    # 比较操作
    print(t1 == t2) # False

    # 排序
    tokens = [t1, t2]
    print(sorted(tokens))