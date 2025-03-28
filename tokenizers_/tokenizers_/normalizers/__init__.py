from .base import NormalizedString, Normalizer
from .unicode import NFD, NFC, NFKC, NFKD, Nmt
from .strip import Strip, StripAccents
from .utils import Sequence, Lowercase


# Prepend = normalizers.Prepend
# Precompiled = normalizers.Precompiled
# Replace = normalizers.Replace
# ByteLevel = normalizers.ByteLevel

NORMALIZERS = {"nfc": NFC, "nfd": NFD, "nfkc": NFKC, "nfkd": NFKD}

def unicode_normalizer_from_str(normalizer: str) -> Normalizer:
    if normalizer not in NORMALIZERS:
        raise ValueError(
            "{} is not a known unicode normalizer. Available are {}".format(normalizer, NORMALIZERS.keys())
        )

    return NORMALIZERS[normalizer]()