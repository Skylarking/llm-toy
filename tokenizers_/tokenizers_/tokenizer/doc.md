NormalizedString、PreTokenizedString、Split、Split、Split关系和作用详解：

---

### **核心类关系图**
```
原始输入 → NormalizedString → PreTokenizedString → Split → Split → Split
                  ↑                  ↑
BytesToCharOffsetConverter ←──────────┘
```

---

### **1. NormalizedString**
**职责**：
- 字符串规范化处理（如统一大小写、Unicode标准化等）
- 维护原始字符串与规范化字符串之间的偏移映射（alignments）
- 支持安全的字符串切片操作（确保不破坏多字节字符）

**关键能力**：
```python
ns = NormalizedString("Café")
ns.lowercase()  # 规范化处理
slice = ns.slice(Range.Normalized(0,3))  # 安全切片
```

---

### **2. PreTokenizedString**
**职责**：
- 管理整个预处理流程的入口
- 维护原始字符串和所有分割后的片段（Split）
- 协调分割→规范化→分词流程

**典型工作流**：
```python
pts = PreTokenizedString("Hello world")
pts.split(split_func)    # 分割
pts.normalize(norm_func) # 规范化
pts.tokenize(tokenizer)  # 分词
encoding = pts.into_encoding() # 最终编码
```

---

### **3. Split**
**职责**：
- 表示被分割后的文本块
- 持有该块的规范化表示（NormalizedString）
- 存储与该块关联的Token列表

**数据结构**：
```python
@dataclass
class Split:
    normalized: NormalizedString  # 规范化后的文本块
    tokens: Optional[List[Token]] # 关联的分词结果
```

---

### **4. Token**
**职责**：
- 表示单个分词结果
- 包含分词在词汇表中的ID、表面形式和偏移量

**不可变特性**：
```python
token = Token(id=100, value="hello", offsets=(0,5))
# token.offsets = (1,2)  # 会引发FrozenInstanceError
```

---

### **5. Encoding**
**职责**：
- 整合所有分词结果的最终输出
- 包含模型需要的结构化数据（ID、注意力掩码等）
- 处理多序列场景（如问答中的问题+上下文）

**关键字段**：
```python
Encoding(
    ids=[100, 101],
    tokens=["hello", "world"],
    offsets=[(0,5), (6,11)],  # 依赖BytesToCharOffsetConverter
    attention_mask=[1, 1]
)
```

---

### **6. BytesToCharOffsetConverter**
**职责**：
- 在字节偏移（Byte Offset）和字符偏移（Char Offset）之间转换
- 解决多字节字符（如中文、emoji）的偏移对齐问题

**转换原理**：
```
原始字节：中(0-3字节) 文(3-6字节)
字符映射：0 ↔ 0-3字节，1 ↔ 3-6字节
转换示例：(0,6) → (0,2) 字符
```

---

### **协作流程**
1. **输入处理**：
   ```python
   raw_text = "Hello 世界"
   ns = NormalizedString(raw_text)  # 创建基础容器
   pts = PreTokenizedString(ns)      # 初始化预处理流程
   ```

2. **分割阶段**：
   ```python
   def split_func(ns):
       return ns.split(" ", REMOVED) # 按空格分割
   pts.split(split_func) # 生成多个Split
   ```

3. **偏移转换**：
   ```python
   converter = BytesToCharOffsetConverter(raw_text)
   token_offsets = (0,5)  # 字节级偏移
   char_offsets = converter.convert(token_offsets) # (0,5)字符
   ```

4. **构建Encoding**：
   ```python
   tokens = [Token(100, "Hello", (0,5)), Token(101, "世界", (6,12))]
   encoding = Encoding.from_tokens(tokens, type_id=0)
   ```

---

### **常见问题对照表**
| 问题现象 | 可能原因 | 解决方案 |
|---------|---------|---------|
| 偏移量错位 | 字节/字符混淆 | 使用BytesToCharOffsetConverter |
| 分词跨字符 | 未正确使用slice | 检查is_char_boundary |
| 编码不一致 | NormalizedString未统一处理 | 确保所有操作通过NormalizedString |

通过理解这些组件的职责和协作关系，可以更好地构建符合多语言需求的分词系统。
