from abc import ABC
from typing import List
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from .talkative import Talkative


class TalkativeBlenderbot(Talkative):
    def __init__(
        self,
        model_name: str,
        contextual: bool = False,
        context_lines: int = 3,
        knowledge: str = "",
    ):
        self.model_name = model_name
        self.contextual = contextual
        self.knowledge = knowledge
        self.context_lines = context_lines
        self.model = None
        self.tokenizer = None
        self.context: List[str] = []

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    def tell(self, message: str) -> str:
        assert self.model is not None, "Model not loaded."
        assert self.tokenizer is not None, "Tokenizer not loaded."

        context_lines = list(self.context) if self.contextual else []
        context_lines.append(message)
        # Cut off the oldest lines.
        if len(context_lines) > self.context_lines:
            context_lines = context_lines[-self.context_lines :]
        self.context = list(context_lines)
        context_lines.insert(0, self.knowledge)

        bot_input_ids = None
        chat_history_ids = None
        for index, line in enumerate(context_lines):  # Only last line for now.
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            new_user_input_ids = self.tokenizer.encode(
                line + self.tokenizer.eos_token,
                return_tensors="pt",
            )

            # append the new user input tokens to the chat history
            bot_input_ids = (
                torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
                if chat_history_ids is not None
                else new_user_input_ids
            )

            if index == 0:
                # Append the context to the chat history and continue.
                continue

            # generated a response while limiting the total chat history to 1000 tokens,
            chat_history_ids = self.model.generate(
                bot_input_ids,
                max_length=1000,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # pretty print last output tokens from bot
        if chat_history_ids is not None and bot_input_ids is not None:
            response = self.tokenizer.decode(
                chat_history_ids[:, bot_input_ids.shape[-1] :][0],
                skip_special_tokens=True,
            )
            if response:
                return response

        return ""

    def clear_context(self) -> bool:
        # No context.
        return True
