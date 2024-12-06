from dataclasses import dataclass

@dataclass
class TuringLLMConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 16
    n_embd: int = 1024
    hidden_size: int = 4096
    norm_eps: float = 1e-5
