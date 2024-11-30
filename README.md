# Turing-LLM Explorer

Welcome to Turing-LLM Explorer! 😄

A mechanistic interpretability tool to understand the internals of Turing-LLM, a large language model.

Please feel free to contact me using my email for any queries or for a chat 🙂: danieljamesdavies12@gmail.com

## How to Use

First, download the files for this app by doing the following:

-   Click the `<> Code` button in this repository.
-   Then on the dropdown, click `Download ZIP`.
-   Once downloaded, unzip the file (probably named `Turing-LLM-Explorer-main.zip`).

To use the interface (front-end):

-   Navigate to `front-end/` in your terminal and type `npm install` to download the front-end's dependencies.
-   Once installed, type `npm run start` to run the app locally.

To run the logic (back-end):

-   Download the following files and place them in their corresponding target locations:
    -   Turing-LLM-1.0-254M can be downloaded here: https://www.kaggle.com/models/danieljamesdavies/turing-llm-1.0-254m. Once downloaded, place `model_1722550239_03986.pt` in `back-end/TuringLLM/`.
    -   Sparse Autoencoders for All Turing-LLM Layers can be downloaded here: https://www.kaggle.com/datasets/danieljamesdavies/turing-llm-sparse-autoencoders. Once downloaded, place the folder `sae/` inside `back-end/SAE/`.
    -   (We're in the process of uploading this dataset) Turing-LLM Explorer Latent Data can be downloaded here: https://www.kaggle.com/datasets/danieljamesdavies/turing-llm-explorer-latent-data. Once downloaded, place the folder `latent_data` in `/back-end`. Overwrite the original `latent_data` file if requested.
-   Navigate to `back-end/` in your terminal and type `pip install -r requirements.txt` to download the back-end's dependencies.
-   Once installed, type `python app.py` (or `python3 app.py`) to run the app locally
