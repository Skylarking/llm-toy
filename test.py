from tokenizers_ import Tokenizer
from tokenizers_.models import BPE
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

from tokenizers_.trainers import BpeTrainer
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

from tokenizers_.pre_tokenizers import Whitespace
tokenizer.pre_tokenizer = Whitespace()

files = [f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
tokenizer.train(files, trainer)

tokenizer.save("data/tokenizer-wiki.json")


