import os
import pickle

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flask import current_app


# See https://huggingface.co/microsoft/DialoGPT-medium

CHAT_HISTORY = "data/chat_history.pkl"
MAX_TOKENS = 1000


def initialize(app):
    app.dialo_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    app.dialo_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")


def chat(user_input: str, clean_history: bool = False) -> str:

    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = current_app.dialo_tokenizer.encode(
        user_input.strip() + current_app.dialo_tokenizer.eos_token, return_tensors="pt"
    )

    # Try to load the last chat history
    chat_history_ids = None
    if os.path.exists(CHAT_HISTORY) and not clean_history:
        with open(CHAT_HISTORY, "rb") as f:
            chat_history_ids = pickle.load(f)

    # append the new user input tokens to the chat history
    bot_input_ids = (
        torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        if chat_history_ids is not None
        else new_user_input_ids
    )

    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = current_app.dialo_model.generate(
        bot_input_ids,
        max_length=MAX_TOKENS,
        pad_token_id=current_app.dialo_tokenizer.eos_token_id,
    )

    # pickle the chat history
    with open(CHAT_HISTORY, "wb") as f:
        pickle.dump(chat_history_ids, f)

    # pretty print last output tokens from bot
    return current_app.dialo_tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1] :][0],
        skip_special_tokens=True,
    )
