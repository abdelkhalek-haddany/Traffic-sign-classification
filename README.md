# Traffic Sign Classification Project

This project focuses on classifying traffic signs using a dataset that contains various types of traffic signs. The project employs data processing, visualization, and CNN - deep learning  to build a classification model.

### Dataset source: 
https://huggingface.co/datasets/haddany/Traffic_sign_classification

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Project Workflow](#project-workflow)
5. [Video Tutorial](#video-tutorial)

## Introduction

This project aims to build a machine learning model capable of classifying traffic signs based on images. It follows a structured workflow that includes data exploration, preprocessing, model training, and evaluation to ensure accurate classification.

## Dataset

The dataset used in this project is available on [Hugging Face](https://huggingface.co/datasets/haddany/Traffic_sign_classification). It contains labeled images of various traffic signs.

## Installation

To run this project, you need Python installed along with the necessary libraries. You can install the required libraries
## Project Workflow

The workflow of this project includes the following steps:

1. **Data Exploration**:
   - Load the dataset and explore its structure.
   - Visualize the distribution of traffic sign classes using charts and graphs to understand the data better.

2. **Data Preprocessing**:
   - Normalize the images to ensure that all pixel values are scaled to the same range, which improves the model's performance.
   - Apply data augmentation techniques such as rotation, flipping, and zooming to increase the diversity of the training data and reduce overfitting.
   - Split the dataset into training and testing sets to evaluate the model's performance on unseen data.

3. **Model Training**:
   - Build and train a deep learning model (e.g., Convolutional Neural Network - CNN) to classify the traffic signs.
   - Use the training data to fit the model and adjust its parameters for better accuracy.
   - Experiment with different architectures, activation functions, and optimizers to improve model performance.

4. **Evaluation**:
   - Assess the model's performance using the testing data.
   - Calculate evaluation metrics such as accuracy, precision, recall, and confusion matrix to determine the model's effectiveness.
   - Fine-tune the model based on evaluation results to optimize its predictions.



## Video Tutorial

Check out the video tutorial for a detailed walkthrough of this project:

[![Watch the video](https://img.youtube.com/vi/oHYM43BKT5E/hqdefault.jpg)](https://www.youtube.com/watch?v=oHYM43BKT5E)

