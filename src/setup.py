import os

from transformers import AutoModelForCausalLM, AutoTokenizer

AutoTokenizer.from_pretrained(os.environ["MODEL"])
AutoModelForCausalLM.from_pretrained(os.environ["MODEL"])
