from flask import Blueprint, request, jsonify
import os
import logging
import torch
import h5py
import pandas as pd
from transformers import AutoTokenizer



# Suppress Tokenizer logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)



get_latent_data_bp = Blueprint('get_latent_data_bp', __name__)

@get_latent_data_bp.route('/api/latent', methods=['GET'])
def get_latent_data():
    layer = int(request.args['layer']) if 'layer' in request.args else 0
    latent = int(request.args['latent']) if 'latent' in request.args else 1
    
    latent_data_path = os.path.abspath(os.path.join(os.getcwd(), './latent_data'))
    if not os.path.exists(latent_data_path):
        return jsonify({ 'message': 'Failure' })
    latent_data_dir_list = os.listdir(latent_data_path)
    
    latent_frequency = get_latent_frequency(layer, latent, latent_data_path, latent_data_dir_list)
    latent_sequences_tokens = get_latent_top_sequences_tokens(layer, latent, latent_data_path, latent_data_dir_list)
    latent_sequences_values = get_latent_top_sequences_values(layer, latent, latent_data_path, latent_data_dir_list)
        
    tokenizer = get_tokenizer()
    top_sequences_list = get_top_sequences_list(latent_sequences_tokens, latent_sequences_values, tokenizer)
    post_from_sequence_latent_data = get_post_from_sequence_latent_data(layer, latent, latent_data_path, latent_data_dir_list, tokenizer)
    
    return jsonify({
        'message': 'Success',
        'latentFrequency': latent_frequency,
        'topSequencesList': top_sequences_list,
        "postFromSequenceLatentData": post_from_sequence_latent_data
    })





def get_latent_frequency(layer, latent, latent_data_path, latent_data_dir_list):
    if "latents_sae_frequencies" not in latent_data_dir_list:
        if "latents_sae_frequencies.pth" not in latent_data_dir_list:
            print("Error: File \"latents_sae_frequencies.pth\" Not Found")
            return None
        else:
            print("Creating latents_sae_frequencies folder...")
            os.makedirs(f"{latent_data_path}/latents_sae_frequencies", exist_ok=True)
            latents_sae_frequencies = torch.load(f"{latent_data_path}/latents_sae_frequencies.pth", weights_only=True).cpu()
            for i in range(latents_sae_frequencies.shape[0]):
                torch.save(latents_sae_frequencies[i].detach().clone(), f"{latent_data_path}/latents_sae_frequencies/latents_sae_frequencies_l{i}.pth")
            del latents_sae_frequencies
        
    latents_layer_sae_frequencies = torch.load(f"{latent_data_path}/latents_sae_frequencies/latents_sae_frequencies_l{layer}.pth", weights_only=True).cpu()
    latent_frequency = latents_layer_sae_frequencies[latent].item()
    del latents_layer_sae_frequencies
    return latent_frequency





def get_latent_top_sequences_tokens(layer, latent, latent_data_path, latent_data_dir_list):
    if "latents_sae_tokens_from_sequence.h5" not in latent_data_dir_list:
        print("Error: File \"latents_sae_tokens_from_sequence.h5\" Not Found")
    with h5py.File(f"{latent_data_path}/latents_sae_tokens_from_sequence.h5", 'r') as h5f:
        latent_sequences_tokens = torch.from_numpy(h5f['tensor'][layer, latent, :, :])
    return latent_sequences_tokens





def get_latent_top_sequences_values(layer, latent, latent_data_path, latent_data_dir_list):
    if "latents_sae_values_from_sequence.h5" not in latent_data_dir_list:
        print("Error: File \"latents_sae_values_from_sequence.h5\" Not Found")
    with h5py.File(f"{latent_data_path}/latents_sae_values_from_sequence.h5", 'r') as h5f:
        latent_sequences_values = torch.from_numpy(h5f['tensor'][layer, latent, :, :])
    return latent_sequences_values





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
        
        remaining_result_string = tokenizer.decode(latent_sequences_tokens[i].tolist()[1:])
        
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





def get_post_from_sequence_latent_data(layer, latent, latent_data_path, latent_data_dir_list, tokenizer):
    if "unembedding_frequencies" not in latent_data_dir_list:
        print("Error: Folder \"unembedding_frequencies\" Not Found")
        return {}
    if "top_other_latents" not in latent_data_dir_list:
        print("Error: Folder \"top_other_latents\" Not Found")
        return {}
    if "latents_sae_frequencies" not in latent_data_dir_list:
        print("Error: File \"latents_sae_frequencies.pth\" Not Found")
        return {}
    
    layer_unembed_token_frequencies = []
    output_token_frequencies = []
    
    def decode_tokens(tokens):
        return tokenizer.decode(tokens)
    
    # Find Folder & Get layer_unembed_token_frequencies
    unembed_folder_path = False
    for folder in os.listdir(latent_data_path + "/unembedding_frequencies"):
        curr_folder_path = latent_data_path + "/unembedding_frequencies/" + str(folder)
        with h5py.File(f'{curr_folder_path}/latents_topk_layer_unembed_token_frequencies.h5', 'r') as h5f:
            if f"{layer}-{latent}" in h5f:
                layer_unembed_token_frequencies = pd.DataFrame(h5f[f"{layer}-{latent}"][:], columns=["token", "frequency"])
                layer_unembed_token_frequencies['decoded_token'] = layer_unembed_token_frequencies['token'].apply(decode_tokens)
                layer_unembed_token_frequencies = layer_unembed_token_frequencies.to_dict(orient='records')
                unembed_folder_path = curr_folder_path
                break
    if unembed_folder_path is False:
        return {}

    # Get output_token_frequencies
    with h5py.File(f'{unembed_folder_path}/latents_topk_output_token_frequencies.h5', 'r') as h5f:
        output_token_frequencies = pd.DataFrame(h5f[f"{layer}-{latent}"][:], columns=["token", "frequency"])
        output_token_frequencies['decoded_token'] = output_token_frequencies['token'].apply(decode_tokens)
        output_token_frequencies = output_token_frequencies.to_dict(orient='records')
    
    top_other_latents = {}
    top_other_latents_h5_filenames = [
        # "latents_other_sae_latent_indices_avg_sequence_non_adj",
        # "latents_other_sae_latent_indices_top_token_non_adj",
        "latents_other_sae_latent_indices_avg_sequence_adj",
        "latents_other_sae_latent_indices_top_token_adj"
    ]
    
    num_layers = 12
    sae_dim = 40960
    top_frequency_threshold = 2000000
    latents_sae_frequencies = torch.load(f"{latent_data_path}/latents_sae_frequencies.pth", weights_only=True).cpu()
    for top_other_latents_h5_filename in top_other_latents_h5_filenames:
        top_other_latents[top_other_latents_h5_filename] = [False for _ in range(num_layers)]
        top_other_latents[top_other_latents_h5_filename + "_rare"] = [[False for _ in range(80)] for _ in range(num_layers)]
        h5_file = h5py.File(f'{latent_data_path}/top_other_latents/0/{top_other_latents_h5_filename}.h5', 'r')
        for layer_index in range(num_layers):
            try:
                top_other_latents[top_other_latents_h5_filename][layer_index] = torch.tensor(h5_file[f"{layer}-{latent}-{layer_index}"][:])
            except:
                other_latents_dir_list = os.listdir(latent_data_path + "/top_other_latents")
                top_other_latents[top_other_latents_h5_filename][layer_index], h5_file = get_top_other_latents_layer(top_other_latents_h5_filename, layer, latent, layer_index, 0, latent_data_path, other_latents_dir_list)
            top_other_latents[top_other_latents_h5_filename][layer_index] = top_other_latents[top_other_latents_h5_filename][layer_index] + (sae_dim / 2)
            top_other_latents[top_other_latents_h5_filename][layer_index] = top_other_latents[top_other_latents_h5_filename][layer_index].tolist()
            
            for i in range(len(top_other_latents[top_other_latents_h5_filename][layer_index])):
                def filterFunction(latent_index):
                    if latents_sae_frequencies[layer_index][int(latent_index)] >= top_frequency_threshold:
                        return False
                    return True
                top_other_latents[top_other_latents_h5_filename + "_rare"][layer_index][i] = list(filter(filterFunction, top_other_latents[top_other_latents_h5_filename][layer_index][i]))
                
    del latents_sae_frequencies

    return {
        "layerUnembedTokenFrequencies": layer_unembed_token_frequencies,
        "outputTokenFrequencies": output_token_frequencies,
        "topOtherLatents": top_other_latents
    }



def get_top_other_latents_layer(top_other_latents_h5_filename, layer, latent, layer_index, folder_path_other_latents_number, latent_data_path, other_latents_dir_list):
    try:
        folder_path_other_latents = latent_data_path + "/top_other_latents/" + str(folder_path_other_latents_number)
        h5_file = h5py.File(f'{folder_path_other_latents}/{top_other_latents_h5_filename}.h5', 'r')
        tensor = torch.tensor(h5_file[f"{layer}-{latent}-{layer_index}"][:])
    except:
        if str(folder_path_other_latents_number+1) in other_latents_dir_list:
            tensor, h5_file_new = get_top_other_latents_layer(top_other_latents_h5_filename, layer, latent, layer_index, folder_path_other_latents_number+1, latent_data_path, other_latents_dir_list)
            h5_file = h5_file_new
        else:
            tensor = None
    return tensor, h5_file
