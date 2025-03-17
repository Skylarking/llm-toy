from ..base import Model

class BPE(Model):
    def __init__(
        self,
        vocab=None,
        merges=None,
        cache_capacity=None,
        dropout=None,
        unk_token=None,
        continuing_subword_prefix=None,
        end_of_word_suffix=None,
        fuse_unk=None,
        byte_fallback=False,
        ignore_merges=False,
    ):
        self.vocab = vocab
        self.merges = merges
        self.cache_capacity = cache_capacity
        self.dropout = dropout
        self.unk_token = unk_token
        self.continuing_subword_prefix = continuing_subword_prefix
        self.end_of_word_suffix = end_of_word_suffix
        self.fuse_unk = fuse_unk
        self.byte_fallback = byte_fallback
        self.ignore_merges = ignore_merges

    @staticmethod
    def from_file(cls, vocab, merge, **kwargs):
        raise NotImplementedError

    @staticmethod
    def read_file(self, vocab, merges):
        with open(vocab, encoding="utf-8") as f:
            buffer = f.read()
        json = ""


    def get_trainer(self):
        pass

    def id_to_token(self, id):
        pass

    def save(self, folder, prefix):
        pass

    def token_to_id(self, tokens):
        pass

    def tokenize(self, sequence):
        pass


