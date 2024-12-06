from flask import Blueprint, request, jsonify
import os
import time
import pickle
import logging
import torch
from transformers import AutoTokenizer
# from sentence_transformers import SentenceTransformer, util



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
    results = []
    search_words = [keywords.strip().lower() for keywords in search_prompt.split(",")]
    frequencies_on_all_keywords = True
    
    tokenizer = get_tokenizer()
    
    os.makedirs(f"{latent_data_path}/decoded_sequences", exist_ok=True)
    decoded_dir_list = os.listdir(f"{latent_data_path}/decoded_sequences")

    for layer_index in range(12):
        if f"layer_{layer_index}.pkl" not in decoded_dir_list:
            print("Error: File \"decoded_sequences/layer_{layer_index}.pkl\" Not Found")
            return []

    for layer_index in range(12):
        start_time = time.time()
        print(f"Searching Layer {layer_index+1}...", end="\r")
        
        with open(f"{latent_data_path}/decoded_sequences/layer_{layer_index}.pkl", "rb") as f:
            decoded_sequences = pickle.load(f)
            
        search_words_frequencies = torch.zeros((40960), dtype=torch.int16)
        
        for i in range(0, len(decoded_sequences), 10):
            latent_top_sequences = decoded_sequences[i:i + 10]
            for latent_top_sequence in latent_top_sequences:
                if frequencies_on_all_keywords is False or all(search_word.lower() in latent_top_sequence.lower() for search_word in search_words):
                    for search_word in search_words:
                        search_words_frequencies[i//10] += latent_top_sequence.lower().count(search_word.lower())
                
        sorted_indices = torch.argsort(search_words_frequencies, descending=True)
        
        for latent_index in sorted_indices[:4]:
            new_result = { "layer": layer_index, "latent": latent_index.item(), "relevance": search_words_frequencies[latent_index].item(), "topSequencePreviews": [] }
            new_result["topSequencePreviews"] = [{ "decoded": decoded_sequences[latent_index*10+top_sequence_index] } for top_sequence_index in range(4)]
            results.append(new_result)
            
        duration = time.time() - start_time
        print(f"Searched Layer {str(layer_index+1).zfill(len(str(12)))}  |  Duration: {duration:.2f}s  |  Top Latents: {', '.join([str(str(i.item() + 1) + ' (f' + str(search_words_frequencies[i].item()) + ')') for i in sorted_indices[:6]])}")
        
    results = [result for result in results if result["relevance"] > 0]
        
    def sortResultsFunction(result):
        return result["relevance"]
    results.sort(reverse=True, key=sortResultsFunction)
    
    return results
    
    
    
    
# def get_sequence_embedding_results(search_prompt, latent_data_path):
#     results = []
    
#     # Sentence Encoder Model
#     # text_encoder_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#     # text_encoder_model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v3')
#     # text_encoder_model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
#     text_encoder_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
#     text_encoder_model = text_encoder_model.to(device)
#     print("Search Prompt: ", search_prompt)
#     search_prompt_embedding = text_encoder_model.encode(search_prompt, device=device)
#     search_prompt_embedding = torch.tensor(search_prompt_embedding, device=device)
#     embedding_dim = search_prompt_embedding.shape[-1]
#     del text_encoder_model
#     torch.cuda.empty_cache()
    
#     tokenizer = get_tokenizer()
    
#     sequence_embeddings_path = latent_data_path + "/sequence_embeddings_all-mpnet-base-v2"
#     latent_data_sequence_embedding_dir_list = os.listdir(sequence_embeddings_path)
#     for layer_index in range(12):
#         name = f"sequence_embeddings_layer_{layer_index}"
#         if f"{name}.pt" not in latent_data_sequence_embedding_dir_list:
#             if f"{name}.h5" not in latent_data_sequence_embedding_dir_list:
#                 return jsonify({ 'message': f'File Not Found: {name}.h5' })
#             print(f"Creating {sequence_embeddings_path}/{name}.pt...")
#             with h5py.File(f"{sequence_embeddings_path}/{name}.h5", 'r') as hdf5_file:
#                 dataset_list = [torch.tensor(dataset[()]) for key, dataset in hdf5_file.items()]
#             sequence_embeddings = torch.cat(dataset_list, dim=0).view(40960, 10, embedding_dim).to(device)
#             torch.save(sequence_embeddings, f"{sequence_embeddings_path}/{name}.pt")
#         else:
#             sequence_embeddings = torch.load(f"{sequence_embeddings_path}/{name}.pt").to(device)
    
#         cosine_similarities = util.cos_sim(search_prompt_embedding, sequence_embeddings.view(-1, embedding_dim))[0].view(40960, 10)
#         # latent_cosine_values, _ = cosine_similarities.max(dim=1)
#         latent_cosine_values = cosine_similarities.mean(dim=1)
#         sorted_indices = torch.argsort(latent_cosine_values, descending=True)
#         if layer_index == 0:
#             print(cosine_similarities[1], sorted_indices.tolist().index(1))
        
#         for latent_index in sorted_indices[:8]:
#             new_result = { "layer": layer_index, "latent": latent_index.item(), "relevance": latent_cosine_values[latent_index].item(), "topSequencePreviews": [] }
#             with h5py.File(f"{latent_data_path}/latents_sae_tokens_from_sequence.h5", 'r') as h5f:
#                 topSequencesTokens = [sequence.tolist()[1:] for sequence in np.asarray(h5f['tensor'][layer_index, latent_index, :, :])]
#             with h5py.File(f"{latent_data_path}/latents_sae_values_from_sequence.h5", 'r') as h5f:
#                 topSequencesValues = [sequence.tolist()[1:] for sequence in np.asarray(h5f['tensor'][layer_index, latent_index, :, :])]
#             top_sequences_list = get_top_sequences_list(torch.tensor(topSequencesTokens[:5]), torch.tensor(topSequencesValues[:5]), tokenizer)
#             for top_sequence_list in top_sequences_list:
#                 new_result["topSequencePreviews"].append({ "tokens": [], "values": [] })
#                 new_result["topSequencePreviews"][-1]["tokens"] = [pairs[0] for pairs in top_sequence_list[1:]]
#                 new_result["topSequencePreviews"][-1]["values"] = [pairs[1] for pairs in top_sequence_list[1:]]
#             results.append(new_result)
#         print(f"Searched Layer {layer_index+1} / 12")
        
        
#     def sortResultsFunction(result):
#         return result["relevance"]

#     results.sort(reverse=True, key=sortResultsFunction)
        
#     return results




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
