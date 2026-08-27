# Sequence Generating Model (Mini Project)

This is a sequence generating language model that is trained on a custom dataset and can generate text that is similar to the training data (Stored in ./Data/data.txt) Can be used to Mimic the text style of an Author or User , as long as enough Data is provided.

## Goal
The goal of this project was to Implement the Decoder-only Transformer Architecture from the 2017 Paper "Attention is All you Need" from scratch.

## Features

- **Tokenizer**: Converts text to integers and integers to text
- **Dataset**: Loads the training data from a file
- **Decoder**: A transformer with Masked self-attention Blocks that predicts the next token in a sequence.



## Prerequisites

- Python 3.13.14
- PyTorch 2.13.0

