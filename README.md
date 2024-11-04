# Multi-layer Perceptron Network for Image Classification

## Overview

This project involves building and training a Multi-layer Perceptron (MLP) and Softmax classifier for image classification tasks on the STL-10 dataset. The MLP is configured to classify images into one of ten categories, focusing on implementing and optimizing key components of multi-layer neural networks.

## Project Objectives

- **Image Dataset Preprocessing**: Experience with large image datasets, including resizing, normalization, and partitioning.
- **Gradient Descent Variants**: Explore and compare Batch Gradient Descent (BGD), Stochastic Gradient Descent (SGD), and Mini-batch Gradient Descent (MBGD).
- **Regularization Techniques**: Apply regularization to reduce overfitting and improve generalization.
- **Multi-class Classification**: Implement and evaluate softmax classification.
- **Hyperparameter Tuning**: Perform grid and random search to optimize hyperparameters.
- **Visualization**: Visualize network weights to understand learned features.

## Project Structure

### Notebooks and Code Files

- **preprocess_stl10.ipynb**: Preprocesses STL-10 images for training.
- **softmax_layer.ipynb**: Contains the implementation of a single-layer Softmax classifier.
- **mlp.ipynb**: Implements and tests a multi-layer perceptron (MLP) for multi-class classification.
- **preprocess_data.py**: Contains functions for loading, normalizing, and creating training/validation/test splits.
- **mlp.py**: Defines the MLP class with forward and backward passes, using ReLU and softmax activations.
- **softmax.py**: Contains the SoftmaxLayer class for single-layer softmax classification.
- **load_stl10_dataset.py**: Downloads and loads the STL-10 dataset.

### Data Files

- **cis_train.dat** and **cis_test.dat**: Synthetic "Circle in a Square" dataset.
- **class_names.txt**: Class labels for STL-10.
- **STL-10 Dataset**: Used for training the MLP and Softmax classifiers.
