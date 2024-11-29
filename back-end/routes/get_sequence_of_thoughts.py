from flask import Blueprint, request, jsonify
from dataclasses import dataclass
import os
import h5py
import torch
import numpy as np
import pickle
from collections import Counter
from TuringLLM.inference import TuringLLMForInference
from SAE.SAE_TopK import SAE
import time
from concurrent.futures import ThreadPoolExecutor



@dataclass
class TuringLLMConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 16
    n_embd: int = 1024
    hidden_size: int = 4096
    norm_eps: float = 1e-5
    
    
    
sae_dim = 40960



get_sequence_of_thoughts_bp = Blueprint('get_sequence_of_thoughts_bp', __name__)

@get_sequence_of_thoughts_bp.route('/api/get-sequence-of-thoughts', methods=['POST'])
def get_sequence_of_thoughts():
    
    data = request.get_json()
    tokenIds = data['tokenIds'] if 'tokenIds' in data else []
    tokenIndex = data['tokenIndex']+3 if 'tokenIndex' in data else -1
    if len(tokenIds) == 0 or tokenIndex == -1 or tokenIndex > len(tokenIds):
        return jsonify({ 'message': 'Invalid Inputs', 'error': True })
    
    torch.cuda.empty_cache()
    
    tokens = tokenIds[:tokenIndex]
    max_length = len(tokens)+1
    
    latent_data_path = os.path.abspath(os.path.join(os.getcwd(), './latent_data'))
    if not os.path.exists(latent_data_path):
        return jsonify({ 'message': 'Failure' })
    latent_data_dir_list = os.listdir(latent_data_path)
    
    sae_path = os.path.abspath(os.path.join(os.getcwd(), './SAE/sae'))
    if not os.path.exists(sae_path):
        return jsonify({ 'message': 'Failure' })
    
    print("Getting Sequences of Thoughts  |  Turing Inference...", end="\r")
    turing_llm = TuringLLMForInference(max_length=max_length, collect_latents=True)
    results, latents = turing_llm.generate_batch([tokens], max_length=max_length, tokenize=False)
    turing_llm.clear()
    
    
    latents_sae_frequencies = torch.load(f"{latent_data_path}/latents_sae_frequencies.pth", weights_only=True).cpu()
    top_frequency_threshold = 2000000
    latents_sae_frequencies_mask = (latents_sae_frequencies < top_frequency_threshold).int()
    del latents_sae_frequencies
    
    top_latents = [[] for _ in range(12)]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    sae_models = {}
    sae_inputs = {}
    sae_load_start_time = time.time()
    def load_model(i, sae_path):
        sae_model = SAE(TuringLLMConfig.n_embd, sae_dim, 128, only_encoder=True).to(device)
        sae_model.load(f"{sae_path}/sae_layer_{i}.pth")
        if i == 0:
            sae_model.k = 128 + (4 * 16)
        else:
            sae_model.k = 128 + ((i) * 16)
        sae_models[str(i)] = torch.compile(sae_model)
        sae_inputs[str(i)] = latents[0][i][0][-1].to(device)
        return sae_model
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(load_model, i, sae_path)
            for i in range(len(latents[0]))
        ]
        for future in futures:
            future.result()
    
    print(f"Load SAE Duration: {time.time()-sae_load_start_time:.2f}s                                                     ")
    
    if device == "cuda":
        batch_size = min(12, len(latents[0]))

        all_top_indices = {}
        all_top_values = {}
        for i in range(0, len(latents[0]), batch_size):
            streams = [torch.cuda.Stream(device=device) for _ in range(batch_size)]
            for j in range(batch_size):
                with torch.cuda.stream(streams[j]):
                    layer_sae_latents, _, _ = sae_models[str(i+j)].encode(sae_inputs[str(i+j)])
                    layer_sae_latents = layer_sae_latents * latents_sae_frequencies_mask[i+j].to(device)
                    top_values, top_indices = torch.topk(layer_sae_latents, 7)
                    all_top_indices[str(i+j)] = top_indices
                    all_top_values[str(i+j)] = top_values
            torch.cuda.synchronize()
              
        sae_decoded_start_time = time.time()
        def get_top_latents(i, latent_data_path, all_top_indices, all_top_values):
            with open(f"{latent_data_path}/decoded_sequences/layer_{i}.pkl", "rb") as f:
                decoded_sequences = pickle.load(f)
            for index, value in zip(all_top_indices[str(i)], all_top_values[str(i)]):
                top_latents[i].append({ "latent": int(index), "value": float(value), "topSequences": decoded_sequences[(index*10):(index*10)+2] })
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(get_top_latents, i, latent_data_path, all_top_indices, all_top_values)
                for i in range(len(latents[0]))
            ]
            for future in futures:
                future.result()
        print(f"SAE Decoded Duration: {time.time()-sae_decoded_start_time:.2f}s                                                     ")
        
    else:
        sae_model = SAE(TuringLLMConfig.n_embd, sae_dim, 128, only_encoder=True).to(device)
        sae_model = torch.compile(sae_model)
        for layer_index, layer_latents in enumerate(latents[0]):
            print(f"Getting Sequences of Thoughts  |  Layer {layer_index+1}  |  Processing...", end="\r")
            sae_model.load(f"{sae_path}/sae_layer_{layer_index}.pth")
            if layer_index == 0:
                sae_model.k = 128 + (4 * 16)
            else:
                sae_model.k = 128 + (layer_index * 16)
            layer_latents = layer_latents[0][-1].to(device)
            layer_sae_latents, _, _ = sae_model.encode(layer_latents)
            layer_sae_latents = layer_sae_latents * latents_sae_frequencies_mask[layer_index].to(device)
            top_values, top_indices = torch.topk(layer_sae_latents, 7)
            with open(f"{latent_data_path}/decoded_sequences/layer_{layer_index}.pkl", "rb") as f:
                decoded_sequences = pickle.load(f)
            for index, value in zip(top_indices, top_values):
                top_latents[layer_index].append({ "latent": int(index), "value": float(value), "topSequences": decoded_sequences[(index*10):(index*10)+2] })
        del sae_model
    torch.cuda.empty_cache()
    
    top_latents_indices = [[item["latent"] for item in top_latents_layer] for top_latents_layer in top_latents]

    latent_relationships = Counter()
    top_other_latents_dir_list = [folder for folder in os.listdir(latent_data_path + "/top_other_latents") if ".txt" not in folder]
    for folder_index, post_from_sequence_latent_data_folder in enumerate(top_other_latents_dir_list):
        print(f"Getting Sequences of Thoughts  |  Folder {folder_index+1}  |  Processing...          ", end="\r") # latents_other_sae_latent_indices_avg_sequence_adj || latents_other_sae_latent_indices_top_token_adj
        other_latents_h5_file = h5py.File(f"{latent_data_path}/top_other_latents/{post_from_sequence_latent_data_folder}/latents_other_sae_latent_indices_avg_sequence_adj.h5")
        for top_latents_layer_index, top_latents_layer in enumerate(top_latents):
            print(f"Getting Sequences of Thoughts  |  Folder {folder_index+1}  |  Processing Layer {top_latents_layer_index+1}...          ", end="\r")
            for latent_dict in top_latents_layer:
                for layer_index in range(12):
                    try:
                        other_sae_latent_indices = np.expand_dims(other_latents_h5_file[f'{top_latents_layer_index}-{latent_dict["latent"]}-{layer_index}'], axis=0)
                        other_sae_latent_indices = other_sae_latent_indices + (sae_dim / 2)
                        other_sae_latent_indices = other_sae_latent_indices.flatten().astype(int)
                        if layer_index == top_latents_layer_index:
                            other_sae_latent_indices = other_sae_latent_indices[other_sae_latent_indices != latent_dict["latent"]]
                        unique_items, counts = np.unique(other_sae_latent_indices, return_counts=True)
                        for item, count in zip(unique_items, counts):
                            if item in top_latents_indices[layer_index]:
                                latent_relationships[((top_latents_layer_index, latent_dict["latent"]), (layer_index, item))] += int(count)
                    except:
                        print("", end="")
        other_latents_h5_file.close()
    latent_relationships = [
        [[[int(item[0][0][0]), int(item[0][0][1])], [int(item[0][1][0]), int(item[0][1][1])]], int(item[1])]
        for item in list(latent_relationships.items())
    ]
    latent_relationships = sorted(latent_relationships, key=lambda item: item[1], reverse=True)
    
    print("Gathered Sequences of Thoughts                                                                                                  ")
    
    return jsonify({ 'success': True, 'message': 'Success', "tokens": tokens, "top_latents": top_latents, "latent_relationships": latent_relationships })
