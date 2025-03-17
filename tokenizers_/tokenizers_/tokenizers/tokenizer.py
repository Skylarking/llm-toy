from typing import List

class Tokenizer:
    def __init__(self, model=None):
        pass

    def from_model(self, model):
        raise NotImplementedError()

    def from_pretrained(self, model):
        raise NotImplementedError()

    def train(self, files: List[str]):
        pass