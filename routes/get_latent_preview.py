from flask import Blueprint, request, jsonify
import os
import logging
import torch
import h5py
from transformers import AutoTokenizer



# Suppress Tokenizer logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)



get_latent_preview_bp = Blueprint('get_latent_preview_bp', __name__)

@get_latent_preview_bp.route('/api/latent/preview', methods=['GET'])
def get_latent_preview():
    layer = int(request.args['layer']) if 'layer' in request.args else 0
    latent = int(request.args['latent']) if 'latent' in request.args else 1
    
    latent_data_path = os.path.abspath(os.path.join(os.getcwd(), './latent_data'))
    if not os.path.exists(latent_data_path):
        return jsonify({ 'message': 'Failure' })
    latent_data_dir_list = os.listdir(latent_data_path)
    
    if "latents_sae_tokens_from_sequence.h5" not in latent_data_dir_list:
        print("Get Latent Data  |  Error: File \"latents_sae_tokens_from_sequence.h5\" Not Found")
    with h5py.File(f"{latent_data_path}/latents_sae_tokens_from_sequence.h5", 'r') as h5f:
        latent_sequences_tokens = torch.from_numpy(h5f['tensor'][layer, latent, :3, 1:])
    
    tokenizer = get_tokenizer()
    top_sequences_list = tokenizer.batch_decode([tokens.tolist() for tokens in latent_sequences_tokens])
    
    return jsonify({ 'success': True, 'message': 'Success', "topSequences": top_sequences_list })



def get_tokenizer():
    tokenizer_model_id = "microsoft/Phi-3-mini-4k-instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id, use_fast=True, local_files_only=True, _fast_init=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id, use_fast=True, _fast_init=True)
    return tokenizer
