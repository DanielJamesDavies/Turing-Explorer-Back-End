from flask import Blueprint, request, jsonify, current_app, Response
import time
import torch
import json
import uuid
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
    max_length = len(tokens)+2+(800 if current_app.config["device"] == "cuda" else 128)
    
    torch.cuda.empty_cache()
    
    inference_id = str(uuid.uuid4())
    
    print("Inference  |  Running Turing-LLM...", end="\r")
    start_time = time.time()
    def generate(turing_llm):
        generation_stream = turing_llm.generate_stream(tokens, max_length=max_length, tokenize=False)
        i = 0
        for response_decoded, response_tokens, _, final  in generation_stream:
            response_tokens_decoded = get_top_sequences_list(response_decoded, response_tokens[2:], tokenizer)
            print(f"Inference  |  Running Turing-LLM...  | Tokens: {len(response_tokens)}", end="\r")
            if final is True:
                print(f"Inference  |  Completed Running Turing-LLM  |  Tokens: {max_length}  |  Duration: {time.time()-start_time:.2f}s")
            yield json.dumps({ 'message': 'Success', 'first': i == 0, 'final': final, "inference_id": inference_id, "response_tokens": response_tokens, "response_tokens_decoded": response_tokens_decoded }) + "<|END_OF_RESPONSE_CHUNK_12|>"
            i += 1
    
    return Response(generate(current_app.config["turing_llm"]), mimetype='application/json')



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
