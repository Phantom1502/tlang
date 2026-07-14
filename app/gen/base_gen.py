class BaseGenerator:
    def __init__(self, tokenizer=None, batch_size: int = 2000):
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError("Subclasses must implement __next__")