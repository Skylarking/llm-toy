from .base import Trainer

class BpeTrainer(Trainer):
    def __init__(self):
        super().__init__()
        self.vocab_size = 0
        self.special_tokens = None


