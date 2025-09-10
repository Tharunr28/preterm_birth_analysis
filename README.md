# preterm_birth_analysis
# Federated Learning for Preterm Birth Prediction

This project implements a **Federated Learning system** for predicting preterm birth using clinical features. It allows multiple clients (hospitals or institutions) to train local models on their own data while maintaining privacy. A central server aggregates the local models to create a global model without sharing raw data.

---

## Features

- **Federated Learning Setup:** Train models locally and aggregate on a central server.
- **Privacy-Preserving:** Raw client data never leaves the client.
- **Binary Classification:** Predicts the probability of preterm birth.
- **NaN Handling:** Automatically handles missing values in datasets.
- **Gradio Interface:** Simple web interface for training and predictions.

---

## Project Structure

