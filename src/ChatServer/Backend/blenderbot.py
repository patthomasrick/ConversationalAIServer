from abc import ABC

from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer

from .talkative import Talkative


class TalkativeBlenderbot(Talkative):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.context = None

    def load(self):
        self.tokenizer = BlenderbotTokenizer.from_pretrained(self.model_name)
        self.model = BlenderbotForConditionalGeneration.from_pretrained(self.model_name)

    def tell(self, message: str) -> str:
        assert self.model is not None, "Model not loaded."
        assert self.tokenizer is not None, "Tokenizer not loaded."

        input_tokens = self.tokenizer(message, return_tensors="pt")
        output_tokens = self.model.generate(**input_tokens)
        return self.tokenizer.decode(output_tokens[0])

    def clear_context(self) -> bool:
        # No context.
        return True
