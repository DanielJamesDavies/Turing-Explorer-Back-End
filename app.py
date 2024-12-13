import __main__
from flask import Flask, request, make_response, jsonify
from flask_cors import CORS
from dataclasses import dataclass
import time
import os
import torch
import pandas as pd
import pickle
from concurrent.futures import ThreadPoolExecutor

from routes.get_latent_data import get_latent_data_bp
from routes.get_latent_preview import get_latent_preview_bp
from routes.search_latents import search_latents_bp
from routes.inference import inference_bp
from routes.get_sequence_of_thoughts import get_sequence_of_thoughts_bp

from TuringLLM.inference import TuringLLMForInference
from SAE.SAE_TopK import SAE



app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})



device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = "cpu"
app.config["device"] = device



@dataclass
class TuringLLMConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 16
    n_embd: int = 1024
    hidden_size: int = 4096
    norm_eps: float = 1e-5
__main__.TuringLLMConfig = TuringLLMConfig
print("Loading Turing-LLM...", end="\r")
turing_load_start_time = time.time()
app.config["turing_llm"] = TuringLLMForInference(max_length=3, device=device)
app.config["turing_llm"].generate(app.config["turing_llm"].tokenizer.encode("Turing"), max_length=3, tokenize=False)
print(f"Loaded Turing LLM in {time.time()-turing_load_start_time:.2f}s")


sae_dim = 40960
print("Loading SAEs...", end="\r")
app.config["sae_models"] = {}
sae_load_start_time = time.time()
sae_path = os.path.abspath(os.path.join(os.getcwd(), './SAE/sae'))
if not os.path.exists(sae_path):
    print("Failed to Load SAEs: Folder \"./SAE/sae\" Not Found")
else:
    def load_model(i, sae_path):
        sae_model = SAE(TuringLLMConfig.n_embd, sae_dim, 128, only_encoder=True, device=device).to(device)
        sae_model.load(f"{sae_path}/sae_layer_{i}.pth")
        if i == 0:
            sae_model.k = 128 + (4 * 16)
        else:
            sae_model.k = 128 + ((i) * 16)
        app.config["sae_models"][str(i)] = torch.compile(sae_model)
        return sae_model
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(load_model, i, sae_path)
            for i in range(TuringLLMConfig.n_layer)
        ]
        for future in futures:
            future.result()
    print(f"Loaded SAEs in {time.time()-sae_load_start_time:.2f}s")
    
    
    
    
def get_decoded_sequences():
    try:
        start_time = time.time()
        print("Loading Decoded Sequences...", end="\r")
        os.makedirs("./latent_data/decoded_sequences", exist_ok=True)
        decoded_dir_list = os.listdir("./latent_data/decoded_sequences")
        for layer_index in range(12):
            for i in range(8):
                if f"layer_{layer_index}_batch_{i}.pkl" not in decoded_dir_list:
                    print("Error: File \"decoded_sequences/layer_{layer_index}_batch_{i}.pkl\" Not Found")
                    return None
        def get_layer_results(layer_index, batch_index, all_dfs):
            with open(f"./latent_data/decoded_sequences/layer_{layer_index}_batch_{batch_index}.pkl", "rb") as f:
                layer_decoded_sequences = pickle.load(f)
            latent_offset = batch_index * (40960 // 8)
            latent_range = range(latent_offset, (batch_index+1) * (40960 // 8))
            df = pd.DataFrame({'id': [f"{layer_index}-{i+latent_offset}" for i in latent_range], 'layer': layer_index, 'latent': list(latent_range), 'text': layer_decoded_sequences})
            all_dfs[layer_index][batch_index] = df
        max_workers = (os.cpu_count() or 1)
        all_dfs = [[None for j in range(8)] for i in range(12)]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(get_layer_results, layer_index, batch_index, all_dfs)
                for batch_index in range(8) for layer_index in range(12)
            ]
            for future in futures:
                future.result()
        print(f"Loaded Decoded Sequences in {time.time()-start_time:.2f}s")
        return pd.concat([df for sublist in all_dfs for df in sublist], ignore_index=True)
    except:
        return None
    
app.config["decoded_sequences"] = get_decoded_sequences()



@app.before_request
def handle_options_requests():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response, 200



@app.route("/", methods=["GET"])
def test():
    return jsonify({"message": "Hello! 👋"}), 200



app.register_blueprint(get_latent_data_bp)
app.register_blueprint(get_latent_preview_bp)
app.register_blueprint(search_latents_bp)
app.register_blueprint(inference_bp)
app.register_blueprint(get_sequence_of_thoughts_bp)



if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, port=5000)
