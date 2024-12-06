from flask import Blueprint, request, jsonify, current_app
import torch
from TuringLLM.tokenizer import Tokenizer



inference_bp = Blueprint('inference_bp', __name__)

@inference_bp.route('/api/inference', methods=['POST'])
def inference():
    
    data = request.get_json()
    prompt = str(data['prompt']) if 'prompt' in data else ""
    if len(prompt.lstrip()) == 0:
        return jsonify({ 'message': 'Invalid Inputs', 'error': True })
    
    tokenizer = Tokenizer()
    tokens = tokenizer.encode(prompt)
    if len(tokens) > 1000:
        tokens = tokens[:1000]
    max_length = len(tokens)+2+512
    
    torch.cuda.empty_cache()
    
    response_decoded, response_tokens, _ = current_app.config["turing_llm"].generate(tokens, max_length=max_length, tokenize=False)
    response_tokens_decoded = get_top_sequences_list(response_decoded, response_tokens[2:], tokenizer)
    
    torch.cuda.empty_cache()
    
    return jsonify({ 'message': 'Success', "response_tokens": response_tokens, "response_tokens_decoded": response_tokens_decoded })



def get_top_sequences_list(decoded, tokens, tokenizer):
    tokens_decoded = []     
    remaining_result_string = str(decoded)
    for token in tokens:
        decoded_token = tokenizer.decode(token)
        token = ""
        for character in remaining_result_string:
            if remaining_result_string[:len(decoded_token)] == decoded_token:
                token += decoded_token
                remaining_result_string = remaining_result_string[len(decoded_token):]
                break
            token += character
            remaining_result_string = remaining_result_string[1:]
            
        tokens_decoded.append(token)
    return tokens_decoded
