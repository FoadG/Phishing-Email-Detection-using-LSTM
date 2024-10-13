# Phishing Email Detection using LSTM

This project focuses on building a machine learning model to detect phishing emails using a Long Short-Term Memory (LSTM) neural network. Phishing emails are a major cybersecurity threat, and this model is designed to help automatically classify emails as either phishing or legitimate based on their textual content.

## Dataset

The dataset used for this project comes from [Kaggle](https://www.kaggle.com/subhajournal/phishingemails), which contains labeled email data with both phishing and legitimate emails. The data is preprocessed and split into training, validation, and test sets for model training and evaluation.

## Model Overview

We implemented an LSTM-based neural network using TensorFlow/Keras. The LSTM architecture is well-suited for sequence-based data like text, making it a great choice for this email classification task. The model consists of the following layers:

- **Embedding Layer**: Transforms the text data into dense vectors of fixed size.
- **LSTM Layer**: Captures the sequential patterns in the email text, allowing the model to learn contextual relationships between words.
- **Dense Layer with Sigmoid Activation**: Outputs a binary classification (phishing or legitimate).

## Training Procedure

- **Text Preprocessing**: The email text data was tokenized and converted into sequences of integers. We limited the vocabulary size to the 10,000 most frequent words, and sequences were padded to a maximum length of 50.
- **Data Splitting**: The dataset was split into training, validation, and test sets. The training and validation sets were used during the model training process, while the test set was reserved for final model evaluation.
- **Model Compilation**: The model was compiled using the `Adam` optimizer and `binary_crossentropy` loss function, which is appropriate for binary classification tasks.
- **Training**: The model was trained over 10 epochs with a batch size of 32. During training, we monitored the accuracy on the validation set to prevent overfitting.

## Results

The model was evaluated on the test set to ensure its effectiveness in classifying emails. Accuracy metrics were used to measure performance, and the results demonstrated that the LSTM model can effectively distinguish between phishing and legitimate emails.

## Installation and Usage

To run this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/phishing-email-detection.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle:
   ```bash
   kaggle datasets download -d subhajournal/phishingemails
   ```

4. Run the model training and evaluation code in the provided Jupyter notebook (`textclassification.ipynb`).

## Future Work

In future iterations, the following improvements could be made:
- Experiment with more complex architectures, such as bidirectional LSTMs or transformer models.
- Incorporate additional email metadata, such as sender information or email subject lines, to improve classification accuracy.
- Fine-tune hyperparameters like the sequence length, batch size, and number of LSTM units.

