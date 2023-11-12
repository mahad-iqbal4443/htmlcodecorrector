 Optimized HTML Correction Model

## Overview

The HTML Correction Model is a deep learning-based system designed to correct HTML markup errors and inconsistencies. Its primary goal is to improve the accuracy of HTML markup by taking input HTML text with potential formatting issues and producing corrected HTML output.

## Dataset

The training dataset consists of pairs of input and target HTML strings, where the target represents the corrected version of the input. The dataset covers various HTML elements and formatting styles to ensure the model's versatility.

## Model Architecture

The model utilizes a sequence-to-sequence architecture that includes an embedding layer, bidirectional LSTM layers with batch normalization, and a time-distributed dense layer for output. The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss.

## Training

The model was trained on the provided dataset for 10 epochs with a batch size of 16. However, given the limited size of the dataset, the achieved accuracy may be suboptimal. To enhance the model's accuracy, it is recommended to include additional data in the training set.

## Tokenization

The input and target text tokenization is performed using the Keras Tokenizer. The configurations of the tokenizers used during training are saved for later use during inference.

## Inference

To demonstrate the correction capability of the trained model on new HTML text, an inference script is provided. The script loads the trained model and tokenizers, performs tokenization and prediction, and converts the predicted sequence back to HTML text.

## Instructions

1. Clone the repository: `git clone https://github.com/mahad-iqbal4443/htmlcodecorrector`
2. Install the required libraries: `pip install -r requirements.txt`
3. Run the inference script: `python inference.py`
4. If you get error during execution recheck the input text or just try with jupyternotebook(ipynb) file.

## Note

The model's accuracy may be limited due to the small size of the dataset. To enhance the model, it is recommended to collect and incorporate a more extensive and diverse dataset to improve its generalization capabilities.

Feel free to explore the code, test the model, and provide feedback. Collaboration and further improvements are welcome.

Thank you for considering my work!