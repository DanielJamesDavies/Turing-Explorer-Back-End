from flask import Flask, request, make_response
from flask_cors import CORS
from dataclasses import dataclass

from routes.get_latent_data import get_latent_data_bp
from routes.search_latents import search_latents_bp
from routes.inference import inference_bp
from routes.get_sequence_of_thoughts import get_sequence_of_thoughts_bp


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@dataclass
class TuringLLMConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 16
    n_embd: int = 1024
    hidden_size: int = 4096
    norm_eps: float = 1e-5


@app.before_request
def handle_options_requests():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response, 200


app.register_blueprint(get_latent_data_bp)
app.register_blueprint(search_latents_bp)
app.register_blueprint(inference_bp)
app.register_blueprint(get_sequence_of_thoughts_bp)


app.run(debug=True, threaded=True, port=5000)
