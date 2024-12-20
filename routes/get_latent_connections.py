from flask import Blueprint, request, jsonify, current_app
from dataclasses import dataclass
import os
import h5py
import torch
import numpy as np
from collections import Counter
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



get_latent_connections_bp = Blueprint('get_latent_connections_bp', __name__)

@get_latent_connections_bp.route('/api/get-latent-connections', methods=['POST'])
def get_latent_connections():
    
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
    
    print("Get Latent Connections  |  Turing Inference...", end="\r")
    _, _, latents = current_app.config["turing_llm"].generate(tokens[2:], max_length=max_length, tokenize=False, collect_latents=True)
    
    latents_sae_frequencies = torch.load(f"{latent_data_path}/latents_sae_frequencies.pth", weights_only=True, map_location=torch.device('cpu')).cpu()
    top_frequency_threshold = 2000000
    latents_sae_frequencies_mask = (latents_sae_frequencies < top_frequency_threshold).int()
    del latents_sae_frequencies
    
    top_latents = [[] for _ in range(12)]
    device = current_app.config["device"]
    
    sae_inputs = {}
    def load_model(i, sae_path):
        sae_inputs[str(i)] = latents[0][i][0][-1].to(device)
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(load_model, i, sae_path)
            for i in range(len(latents[0]))
        ]
        for future in futures:
            future.result()
    
    all_top_indices = {}
    all_top_values = {}
    
    sae_inference_start_time = time.time()
    if device == "cuda":
        batch_size = min(12, len(latents[0]))
        for i in range(0, len(latents[0]), batch_size):
            streams = [torch.cuda.Stream(device=device) for _ in range(batch_size)]
            for j in range(batch_size):
                with torch.cuda.stream(streams[j]):
                    layer_sae_latents, _, _ = current_app.config["sae_models"][str(i+j)].encode(sae_inputs[str(i+j)])
                    layer_sae_latents = layer_sae_latents * latents_sae_frequencies_mask[i+j].to(device)
                    top_values, top_indices = torch.topk(layer_sae_latents, 7)
                    all_top_indices[str(i+j)] = top_indices
                    all_top_values[str(i+j)] = top_values
            torch.cuda.synchronize()
    else:
        def get_sae_top_features(layer_index, layer_latents, sae_model, all_top_indices, all_top_values):
            print(f"Get Latent Connections  |  Layer {layer_index+1}  |  Processing...", end="\r")
            layer_latents = layer_latents[0][-1].to(device)
            layer_sae_latents, _, _ = sae_model.encode(layer_latents)
            layer_sae_latents = layer_sae_latents * latents_sae_frequencies_mask[layer_index].to(device)
            top_values, top_indices = torch.topk(layer_sae_latents, 7)
            all_top_indices[str(layer_index)] = top_indices
            all_top_values[str(layer_index)] = top_values
        
        batch_size = min(12, len(latents[0]))
        for i in range(0, len(latents[0]), batch_size):
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(get_sae_top_features, i+j, latents[0][i+j], current_app.config["sae_models"][str(i+j)], all_top_indices, all_top_values)
                    for j in range(batch_size)
                ]
                for future in futures:
                    future.result()
    torch.cuda.empty_cache()
    def get_top_latents(i, latent_data_path, all_top_indices, all_top_values):
        for index, value in zip(all_top_indices[str(i)], all_top_values[str(i)]):
            top_latents[i].append({ "latent": int(index), "value": float(value) })
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(get_top_latents, i, latent_data_path, all_top_indices, all_top_values)
            for i in range(len(latents[0]))
        ]
        for future in futures:
            future.result()
    top_latents_indices = [[item["latent"] for item in top_latents_layer] for top_latents_layer in top_latents]
    print(f"Get SAE Features Duration: {time.time()-sae_inference_start_time:.2f}s                                                     ")
    
    print("Getting Top Latent Connections...")
    latent_conections = Counter()
    start_time_get_top_latent_connections = time.time()
    top_other_latents_dir_list = [folder for folder in os.listdir(latent_data_path + "/top_other_latents") if ".txt" not in folder]
    for folder_index, post_from_sequence_latent_data_folder in enumerate(top_other_latents_dir_list):
        other_latents_h5_file = h5py.File(f"{latent_data_path}/top_other_latents/{post_from_sequence_latent_data_folder}/latents_other_sae_latent_indices_avg_sequence_adj.h5")
        for top_latents_layer_index, top_latents_layer in enumerate(top_latents):
            for latent_dict in top_latents_layer:
                latent_index = latent_dict["latent"]
                for connected_layer_index in range(12):
                    try:
                        other_sae_latent_indices = np.expand_dims(other_latents_h5_file[f'{top_latents_layer_index}-{latent_index}-{connected_layer_index}'], axis=0)
                        other_sae_latent_indices = other_sae_latent_indices + (sae_dim / 2)
                        other_sae_latent_indices = other_sae_latent_indices.flatten().astype(int)
                        if connected_layer_index == top_latents_layer_index:
                            other_sae_latent_indices = other_sae_latent_indices[other_sae_latent_indices != latent_index]
                        unique_items, counts = np.unique(other_sae_latent_indices, return_counts=True)
                        for item, count in zip(unique_items, counts):
                            if item in top_latents_indices[connected_layer_index]:
                                latent_conections[tuple(sorted(((top_latents_layer_index, latent_index), (connected_layer_index, item)), key=lambda x: (x[0], x[1])))] += int(count)
                    except:
                        print("", end="")
        other_latents_h5_file.close()
    latent_conections = [
        { 
            "latents":[
                { "layer": int(item[0][0][0]), "latent": int(item[0][0][1]) },
                { "layer": int(item[0][1][0]), "latent": int(item[0][1][1]) }
            ],
            "frequency": int(item[1])
        }
        for item in list(latent_conections.items())
    ]
    print(f"Get Top Latent Connections Duration: {time.time()-start_time_get_top_latent_connections:.2f}s")
    
    print("Gathered Latent Connections")
    
    return jsonify({ 'success': True, 'message': 'Success', "tokens": tokens, "topLatents": top_latents, "latentConections": latent_conections })
