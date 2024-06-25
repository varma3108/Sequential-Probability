# Sequential Probability Estimation Model Project

This project implements a trigram classifier to categorize sentences into different genres using the Brown corpus from the NLTK library. The classifier processes, stems, and tokenizes sentences, builds trigram models for each genre, and evaluates the performance of the classifier using accuracy, precision, recall, and F1-score metrics.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Features](#features)
- [License](#license)

## Installation

To run this project, you'll need to have Python installed. You can install the required dependencies using pip:


pip install nltk numpy regex


## Usage

1. **Download the Brown Corpus**:
   Ensure the Brown corpus is downloaded using NLTK:

   ```python
   import nltk
   nltk.download('brown')
   ```

2. **Run the Classifier**:
   Execute the script to train the classifier and evaluate its performance:

   ```sh
   python nlp_assignment1_2.py
   ```

## Project Structure

- **Data Preprocessing**:
  - `process_sentence(sentence)`: Processes and stems the sentence, removing punctuation and converting to lower case.
  - `preprocess_data(sentences)`: Preprocesses a list of sentences grouped by topic and builds a vocabulary.

- **Genre Conversion**:
  - `convert_fid_to_genre(fid_sentences, output='dict')`: Converts file IDs to genre labels.

- **Trigram Model Building**:
  - `build_ngram(sentences, n=3)`: Builds n-gram models for sentences.
  - `TrigramClassifier`: Class that handles the training, validation, and testing of the trigram classifier.

- **Evaluation**:
  - `evaluate(mode='train')`: Evaluates the classifier's performance.
  - `confusion_matrix(scores)`: Prints the confusion matrix and calculates accuracy, precision, recall, and F1-score.

## Features

- **Sentence Processing**: Stemming and punctuation removal.
- **Genre Classification**: Classifies sentences into predefined genres using trigram models.
- **Performance Evaluation**: Computes and displays evaluation metrics for the classifier.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
