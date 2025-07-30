# Self-Learning for Mix-Supervised Image Classification

This project is implementation for: ***”Mix-supervised Learning: A Self-Learning Strategy for Facilitating Data Utilization and Robust Representation“***. In this work, inspired by human learning, we propose mixed supervised learning and give a self-learning training phase to solve this challenge. 

🚀 Innovative Training Paradigm:

This project implements a self-learning process for deep learning-based image classification, enabling models to learn from validation data in a label-agnostic manner. Unlike traditional supervised learning, our method allows the model to:

Perceive distribution patterns in validation data without explicit labels. Expand learned data distributions beyond training set boundaries. Enhance model robustness by leveraging knowledge from the entire dataset.

🏷️ Key Features:

Mix-Supervised Learning: Unifies training and validation data utilization. Label-Free Adaptation: Models autonomously infer validation data distributions. Plug-and-Play: Compatible with standard CNN/Transformer architectures.

Quick Start:

Install Dependencies

pip install torch torchvision numpy tqdm

Run Training:

Edit DRC_TCO.py to select your model and dataset, then execute:

python DRC_TCO.py
