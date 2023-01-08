# LanguageTranslationWithRNN

## ©️license & copyright©️:

📧 [nivk99](https://github.com/nivk99) -  Niv Kotek (I.D: 208236315)


📧 [SAEED]() - SAEED ESAWI (I.D: 314830985)

![](https://github.com/tommytracey/AIND-Capstone/blob/master/images/translation.gif)


## ❓What is Natural Language Processing (NLP)❓
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of "understanding" the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.
Challenges in natural language processing frequently involve speech recognition, natural-language understanding, and natural-language generation.
![](https://www.asksid.ai/wp-content/uploads/2021/02/an-introduction-to-natural-language-processing-with-python-for-seos-5f3519eeb8368.png)

## ❓What is deep learning❓
Deep learning is a subset of machine learning, which is essentially a neural network with three or more layers. These neural networks attempt to simulate the behavior of the human brain—albeit far from matching its ability—allowing it to “learn” from large amounts of data. While a neural network with a single layer can still make approximate predictions, additional hidden layers can help to optimize and refine for accuracy.
![](https://cdn.educba.com/academy/wp-content/uploads/2020/01/Deep-Learning.jpg)


## ❓What is machine learning❓
Machine learning is a branch of artificial intelligence  and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.
Machine learning is an important component of the growing field of data science. Through the use of statistical methods, algorithms are trained to make classifications or predictions, and to uncover key insights in data mining projects. 
Machine learning algorithms are typically created using frameworks that accelerate solution development, such as TensorFlow and PyTorch.
![](https://miro.medium.com/max/1400/1*cG6U1qstYDijh9bPL42e-Q.jpeg)

## Goal
In this project, we build a deep neural network that functions as part of a machine translation pipeline. The pipeline accepts English text as input and returns the French translation. The goal is to achieve the highest translation accuracy possible.

## Background

Building the Pipeline:

1. Preprocessing:

A. Load and Examine Data:

The inputs are sentences in English and  the outputs are the corresponding translations in French.
![](https://github.com/tommytracey/AIND-Capstone/blob/master/images/training-sample.png)
![](https://github.com/nivk99/LanguageTranslationWithRNN/blob/main/images/Results_3.jpg)

B. Tokenization:

 convert the text to numerical values. This allows the neural network to perform operations on the input data. For this project, each word and punctuation mark will be given a unique ID.
When we run the tokenizer, it creates a word index, which is then used to convert each sentence to a vector.
![](https://github.com/tommytracey/AIND-Capstone/blob/master/images/tokenizer.png)

C. Padding:
When we feed our sequences of word IDs into the model, each sequence needs to be the same length. To achieve this, padding is added to any sequence that is shorter than the max length.
![](https://github.com/tommytracey/AIND-Capstone/blob/master/images/padding.png)

2. Modeling:

build, train, validation and test the model

Inputs: Each word is encoded as a unique integer that maps to the English dataset vocabulary.

Outputs: The outputs are returned as a sequence of integers which can then be mapped to the French dataset vocabulary

3. Prediction: 

generate specific translations of English to French, and compare the output translations to the ground truth translations


We use Keras for the frontend and TensorFlow for the backend in this project.





## Results
Validation accuracy: 98%

Training time: 20 epochs
![](https://github.com/nivk99/LanguageTranslationWithRNN/blob/main/images/Results.jpg)
![](https://github.com/nivk99/LanguageTranslationWithRNN/blob/main/images/Results_2.jpg)

# Project Starter Code
In case you want to run this project yourself, below is the project starter code.

## Install
    @ Python 3
    @ NumPy
    @ TensorFlow 2.x
    @ Keras 2.x
    @ matplotlib
    @ sklearn
    @ tkinter

## [Clone the project and run the app](https://github.com/nivk99/LanguageTranslationWithRNN.git)




