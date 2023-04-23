# MEIA-PROJ3: AI-Driven Customer Support Ticket Categorization and Sentiment Analysis System

This repository contains the code and resources for the AI-Driven Customer Support Ticket Categorization and Sentiment Analysis System. The system aims to enhance customer support ticket management by categorizing tickets, analyzing customer sentiment, and generating appropriate responses using various deep learning models and algorithms.

## Table of Contents

- [Description](#description)
- [File Structure](#file-structure)
- [Getting Started](#getting-started)
- [License](#license)

## Description

The project is divided into four main modules:

1. Ticket Typification: Deep learning models, such as LSTM and CNN, are compared to categorize tickets based on a hierarchical structure.
2. Sentiment Analysis: Pre-trained models and custom transformers are trained for emotion recognition in customer messages using the GoEmotions dataset.
3. Translation: A translation API is implemented, utilizing pre-trained models from the Hugging Face Transformers library and langdetect for language detection.
4. Text Generation: GPT-3.5-turbo is employed to create personalized and empathetic ticket responses based on the customer's emotions and ticket categorization.

## File Structure

- `Documentation/` - Contains all the documentation related to the project.
- `SentimentAnalysis/` - Includes pre-processing, data visualization, model training, and custom transformer implementation for sentiment analysis.
- `Typification/` - Contains pre-processing, data visualization, and LSTM vs CNN comparison for ticket classification.
- `UI/` - Houses frontend code and APIs, including sentiment analysis, typification, translation, and text generation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
