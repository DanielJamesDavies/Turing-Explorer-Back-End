from flask import Blueprint, request, jsonify
import os
import time
import pickle
import h5py
import logging
import torch
from transformers import AutoTokenizer
import pandas as pd
import concurrent.futures



# Suppress Tokenizer logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)



search_latents_bp = Blueprint('search_latents_bp', __name__)

@search_latents_bp.route('/api/search', methods=['GET'])
def search_latents():
    
    search_prompt = str(request.args['q']) if 'q' in request.args else ""
    if len(search_prompt.strip()) == 0:
        return jsonify({ 'message': 'Invalid Inputs', 'error': True })
    
    latent_data_path = os.path.abspath(os.path.join(os.getcwd(), './latent_data'))
    if not os.path.exists(latent_data_path):
        return jsonify({ 'message': 'Latent Data File Path Not Found' })
    latent_data_dir_list = os.listdir(latent_data_path)
    
    for name in ['latents_sae_tokens_from_sequence.h5']:
        if name not in latent_data_dir_list:
            return jsonify({ 'message': f'File Not Found: {name}' })
    
    # results = get_sequence_embedding_results(search_prompt, latent_data_path)
    results = get_keyword_match_results(search_prompt, latent_data_path)
    torch.cuda.empty_cache()
    
    return jsonify({ 'message': 'Success', "results": results })




def get_keyword_match_results(search_prompt, latent_data_path):
    start_time = time.time()
    results = []
    search_words = [keywords.strip().lower() for keywords in search_prompt.split(",")]
    
    os.makedirs(f"{latent_data_path}/decoded_sequences", exist_ok=True)
    decoded_dir_list = os.listdir(f"{latent_data_path}/decoded_sequences")

    for layer_index in range(12):
        for i in range(8):
            if f"layer_{layer_index}_batch_{i}.pkl" not in decoded_dir_list:
                print("Error: File \"decoded_sequences/layer_{layer_index}_batch_{i}.pkl\" Not Found")
                return []
    
    
    print("Search Latents  |  Searching for Latents...", end="\r")
    max_workers = (os.cpu_count() or 1) * 8
    
    def load_pickle_file(layer_index, batch_index):
        with open(f"{latent_data_path}/decoded_sequences/layer_{layer_index}_batch_{batch_index}.pkl", "rb") as f:
            data = pickle.load(f)
        return layer_index, batch_index, data
    all_decoded_sequences = [[] for _ in range(12)]
    loaded_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [(layer, batch) for layer in range(12) for batch in range(8)]
        future_to_task = {executor.submit(load_pickle_file, layer, batch): (layer, batch) for layer, batch in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            layer_index, batch_index, data = future.result()
            loaded_data[(layer_index, batch_index)] = data
    for layer_index in range(12):
        for batch_index in range(8):
            data = loaded_data.get((layer_index, batch_index))
            if data is not None:
                all_decoded_sequences[layer_index].extend(data)
            else:
                print(f"Search Latents  |  Error: Decoded Sequences Layer {layer_index} Batch {batch_index} Not Found")
                return []
    
    tokenizer = get_tokenizer()
    tokens_from_sequence_h5_file = h5py.File(f"{latent_data_path}/latents_sae_tokens_from_sequence.h5", 'r')
    def search_layer_latents(layer_index, results, all_decoded_sequences, tokens_from_sequence_h5_file):
        df = pd.DataFrame({'text': all_decoded_sequences[layer_index] })
        for i, search_word in enumerate(search_words):
            if i == 0:
                df["relevance"] = df['text'].str.count(search_word)
            else:
                df["relevance"] += df['text'].str.count(search_word)
        search_words_frequencies = torch.tensor(df['relevance'].values, dtype=torch.int16)
        sorted_indices = torch.argsort(search_words_frequencies, descending=True)
        
        sorted_indices_sorted, original_order_indices = torch.sort(sorted_indices[:4])
        latent_sequences_tokens = torch.from_numpy(tokens_from_sequence_h5_file['tensor'][layer_index, sorted_indices_sorted, :3, 1:])[original_order_indices]
        for i, latent_index in enumerate(sorted_indices[:4]):
            new_result = { "layer": layer_index, "latent": latent_index.item(), "relevance": search_words_frequencies[latent_index].item(), "topSequencePreviews": [] }
            new_result["topSequencePreviews"] = [{ "decoded": decoded } for decoded in tokenizer.batch_decode(latent_sequences_tokens[i])]
            results.append(new_result)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(search_layer_latents, layer_index, results, all_decoded_sequences, tokens_from_sequence_h5_file)
            for layer_index in range(12)
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f'Search Latents  |  Error: {e}')
        
    print(f"Search Latents  |  Searched Latents  |  Duration: {time.time()-start_time:.2f}s")
        
    results = [result for result in results if result["relevance"] > 0]
    def sortResultsFunction(result):
        return result["relevance"]
    results.sort(reverse=True, key=sortResultsFunction)
    return results




def get_tokenizer():
    tokenizer_model_id = "microsoft/Phi-3-mini-4k-instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id, use_fast=True, local_files_only=True, _fast_init=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id, use_fast=True, _fast_init=True)
    return tokenizer





def get_top_sequences_list(latent_sequences_tokens, latent_sequences_values, tokenizer):
    top_sequences_list = []
    for i in range(latent_sequences_tokens.shape[0]):
        top_sequences_list.append([])
        
        remaining_result_string = tokenizer.decode(latent_sequences_tokens[i].tolist())
        
        for j in range(latent_sequences_tokens.shape[1]):
            token = latent_sequences_tokens[i][j].item()
            if j != 0:
                decoded_token = tokenizer.decode(token)
                token = ""
                for character in remaining_result_string:
                    if remaining_result_string[:len(decoded_token)] == decoded_token:
                        token += decoded_token
                        remaining_result_string = remaining_result_string[len(decoded_token):]
                        break
                    token += character
                    remaining_result_string = remaining_result_string[1:]
                
            value = latent_sequences_values[i][j].item()
            top_sequences_list[-1].append([token, value])
    return top_sequences_list
