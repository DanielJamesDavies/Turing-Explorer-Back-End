import os
import logging
from transformers import AutoTokenizer

# Suppress Tokenizer logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Tokenizer:
    def __init__(self, model_id = "microsoft/Phi-3-mini-4k-instruct"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=True, _fast_init=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, _fast_init=True)

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def get_eos_token(self):
        return self.tokenizer.eos_token_id

    def get_bos_token(self):
        return self.tokenizer.bos_token_id

    def get_pad_token(self):
        return self.tokenizer.bos_token_id
