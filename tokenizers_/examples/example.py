from tokenizers_ import Tokenizer
from tokenizers_.models import BPE
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

