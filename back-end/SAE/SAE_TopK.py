import random
import torch
import torch.nn as nn



random.seed(12)



@torch.no_grad()
def geometric_median(points, max_iter: int = 200, tol: float = 1e-5):
    guess = points.mean(dim=0)
    prev = torch.zeros_like(guess)

    weights = torch.ones(len(points), device=points.device)

    for _ in range(max_iter):
        prev = guess
        weights = 1 / torch.norm(points - guess, dim=1)
        weights /= weights.sum()
        guess = (weights.unsqueeze(1) * points).sum(dim=0)
        if torch.norm(guess - prev) < tol:
            break

    return guess



class SAE(nn.Module):
    
    def __init__(self, d_model, d_sae, k=100, only_encoder=False):
        super().__init__()
        
        self.d_model = d_model
        self.d_sae = d_sae
        self.k = k

        self.encoder = nn.Linear(d_model, d_sae)
        self.encoder.bias.data.zero_()
        
        if only_encoder is not True:
            self.decoder = nn.Linear(d_sae, d_model, bias=False)
            self.decoder.weight.data = self.encoder.weight.data.clone().T
            self.set_decoder_norm_to_unit_norm()

        self.decoder_bias = nn.Parameter(torch.zeros(d_model))
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if only_encoder is not True:
            self.num_tokens_since_fired = torch.zeros(d_sae, dtype=torch.long, device=self.device)
            self.auxk_alpha = 1 / 32
            self.dead_feature_threshold = 10_000_000
        
        self.only_encoder = only_encoder

    def encode(self, x):
        post_relu_feat_acts = nn.functional.relu(self.encoder(x - self.decoder_bias))
        post_topk = post_relu_feat_acts.topk(self.k, sorted=False, dim=-1)

        top_acts = post_topk.values
        top_indices = post_topk.indices

        buffer = torch.zeros_like(post_relu_feat_acts)
        encoded_acts = buffer.scatter_(dim=-1, index=top_indices, src=top_acts)

        return encoded_acts, top_acts, top_indices

    def load(self, path):
        if self.only_encoder is True:
            self.decoder = nn.Linear(self.d_sae, self.d_model, bias=False)
            self.num_tokens_since_fired = torch.zeros(self.d_sae, dtype=torch.long, device=self.device)
        
        self.load_state_dict(torch.load(path))
        
        if self.only_encoder is True:
            del self.decoder
            del self.num_tokens_since_fired
