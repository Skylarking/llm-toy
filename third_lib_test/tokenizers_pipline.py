# -------------------------------- normalizer ---------------------------------- #

from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents
normalizer = normalizers.Sequence([NFD(), StripAccents()])

a = normalizer.normalize_str("Héllò hôw are ü?")
# "Hello how are u?"
print(a)

# -------------------------------- pre tokenizer ---------------------------------- #
from tokenizers.pre_tokenizers import Whitespace
pre_tokenizer = Whitespace()
pre_tokenizer.pre_tokenize_str("Hello! How are you? I'm fine, thank you.")
# [("Hello", (0, 5)), ("!", (5, 6)), ("How", (7, 10)), ("are", (11, 14)), ("you", (15, 18)),
#  ("?", (18, 19)), ("I", (20, 21)), ("'", (21, 22)), ('m', (22, 23)), ("fine", (24, 28)),
#  (",", (28, 29)), ("thank", (30, 35)), ("you", (36, 39)), (".", (39, 40))]