import logging
import os
import pickle
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flask import current_app


# See https://huggingface.co/microsoft/DialoGPT-medium

CHAT_HISTORY_LINES = 2
CHAT_HISTORY_FILE = "data/chat_history_lines.pkl"
CHAT_CONTEXT = (
    """You are a chat bot. You are in love with William Lambert. You hate Sam."""
)
MAX_TOKENS = 1000

DIALO_TOKENIZER = None
DIALO_MODEL = None


def initialize(app, cli=False):
    if not cli:
        logging.info("Loading tokenizer")
        app.dialo_tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL"])
        logging.info("Loading model")
        app.dialo_model = AutoModelForCausalLM.from_pretrained(os.environ["MODEL"])
    else:
        global DIALO_TOKENIZER, DIALO_MODEL
        logging.info("Loading tokenizer")
        DIALO_TOKENIZER = AutoTokenizer.from_pretrained(os.environ["MODEL"])
        logging.info("Loading model")
        DIALO_MODEL = AutoModelForCausalLM.from_pretrained(os.environ["MODEL"])


def get_model(cli=False):
    if not cli:
        return current_app.dialo_model
    else:
        return DIALO_MODEL


def get_tokenizer(cli=False):
    if not cli:
        return current_app.dialo_tokenizer
    else:
        return DIALO_TOKENIZER


def get_history() -> List[str]:
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "rb") as f:
            return pickle.load(f)
    else:
        return []


def clear_history():
    with open(CHAT_HISTORY_FILE, "wb") as f:
        pickle.dump([], f)


def chat(user_input: str, clean_history: bool = False, cli: bool = False) -> str:
    lines: List[str] = []
    if os.path.exists(CHAT_HISTORY_FILE) and not clean_history:
        with open(CHAT_HISTORY_FILE, "rb") as f:
            old_lines = pickle.load(f)
            lines.extend(old_lines[-min(CHAT_HISTORY_LINES, len(old_lines)) :])
    lines.append(user_input)

    with open(CHAT_HISTORY_FILE, "wb") as f:
        pickle.dump(lines, f)

    lines.insert(0, CHAT_CONTEXT)

    bot_input_ids = None
    chat_history_ids = None
    for index, line in enumerate(lines):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = get_tokenizer(cli).encode(
            line + get_tokenizer(cli).eos_token,
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
        chat_history_ids = get_model(cli).generate(
            bot_input_ids,
            max_length=MAX_TOKENS,
            pad_token_id=get_tokenizer(cli).eos_token_id,
        )

    # pretty print last output tokens from bot
    if chat_history_ids is not None and bot_input_ids is not None:
        response = get_tokenizer(cli).decode(
            chat_history_ids[:, bot_input_ids.shape[-1] :][0],
            skip_special_tokens=True,
        )
        if response:
            return response

    clear_history()
    return "..."
