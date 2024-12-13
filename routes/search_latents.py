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
import swifter
import json



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
    
    print("Search Latents  |  Loading Decoded Sequences...", end="\r")
    start_time_load_decoded_sequences = time.time()
    def get_layer_results(layer_index, batch_index, all_dfs):
        with open(f"{latent_data_path}/decoded_sequences/layer_{layer_index}_batch_{batch_index}.pkl", "rb") as f:
            layer_decoded_sequences = pickle.load(f)
        latent_offset = batch_index * (40960 // 8)
        latent_range = range(latent_offset, (batch_index+1) * (40960 // 8))
        df = pd.DataFrame({'id': [f"{layer_index}-{i+latent_offset}" for i in latent_range], 'layer': layer_index, 'latent': list(latent_range), 'text': layer_decoded_sequences})
        all_dfs[layer_index][batch_index] = df
    max_workers = (os.cpu_count() or 1)
    all_dfs = [[None for j in range(8)] for i in range(12)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(get_layer_results, layer_index, batch_index, all_dfs)
            for batch_index in range(8) for layer_index in range(12)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    df = pd.concat([df for sublist in all_dfs for df in sublist], ignore_index=True)
    print(f"Search Latents  |  Loaded Decoded Sequences  |  Duration: {time.time()-start_time_load_decoded_sequences:.2f}s")
    
    print("Search Latents  |  Getting Latent Relevances...", end="\r")
    start_time_get_latent_relevances = time.time()
    for i, search_word in enumerate(search_words):
        if i == 0:
            df["relevance"] = df['text'].swifter.apply(lambda x: x.count(search_word))
        else:
            df["relevance"] += df['text'].swifter.apply(lambda x: x.count(search_word))
    print(f"Search Latents  |  Got Latent Relevances     |  Duration: {time.time()-start_time_get_latent_relevances:.2f}s")

    df = df.drop(columns=['text']).sort_values('relevance', ascending=False).drop_duplicates(subset='id', keep='first')
    top_50 = df.head(50)
    layer_results = []
    for layer in df['layer'].unique():
        layer_df = df[df['layer'] == layer].head(4)
        layer_results.append(layer_df)
    layer_df = pd.concat(layer_results)
    final_df = pd.concat([top_50, layer_df]).drop_duplicates(subset='id', keep='first')
    results = json.loads(final_df.sort_values('latent', ascending=True).to_json(orient='records'))
    
    tokenizer = get_tokenizer()
    tokens_from_sequence_h5_file = h5py.File(f"{latent_data_path}/latents_sae_tokens_from_sequence.h5", 'r')
    for layer_index in range(12):
        layer_items = [[i, item] for i, item in enumerate(results) if item["layer"] == layer_index]
        latent_sequences_tokens = torch.from_numpy(tokens_from_sequence_h5_file['tensor'][layer_index, [item[1]["latent"] for item in layer_items], :3, 1:])
        for i, item_pair in enumerate(layer_items):
            results[item_pair[0]]["topSequencePreviews"] = [{ "decoded": decoded } for decoded in tokenizer.batch_decode(latent_sequences_tokens[i])]
    
    
        
    print(f"Search Latents  |  Searched Latents          |  Duration: {time.time()-start_time:.2f}s")
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
