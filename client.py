import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import gradio as gr
from model import SimpleNN  # Make sure your model.py defines SimpleNN


# Client Class
class Client:
    def __init__(self, client_id, data_path, global_model_path):
        self.client_id = client_id
        self.data_path = data_path
        self.model = SimpleNN(input_size=14, hidden_size=64)  # Adjust input size as necessary
        self.criterion = nn.BCEWithLogitsLoss()  # Binary classification
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.load_global_model(global_model_path)

    def load_global_model(self, global_model_path):
        try:
            self.model.load_state_dict(torch.load(global_model_path))
            self.model.eval()
            print(f"[Client {self.client_id}] Global model loaded.")
        except Exception as e:
            print(f"[Client {self.client_id}] Error loading global model:", e)

    def load_data(self):
        try:
            # Load CSV
            self.data = pd.read_csv(self.data_path)
            print(f"[Client {self.client_id}] Initial data shape:", self.data.shape)

            # Convert all columns to numeric
            self.data = self.data.apply(pd.to_numeric, errors='coerce')

            # Fill NaNs with column mean
            self.data.fillna(self.data.mean(), inplace=True)

            # Ensure target column exists
            if 'Preterm_Birth' not in self.data.columns:
                raise ValueError("Target column 'Preterm_Birth' missing!")

            # Features and target
            X = self.data.drop(columns=['Preterm_Birth']).values
            y = self.data['Preterm_Birth'].values.astype(float)

            # Normalize features
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            # Convert to tensors
            X_train = torch.tensor(X, dtype=torch.float32)
            y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1)

            return X_train, y_train

        except Exception as e:
            print(f"[Client {self.client_id}] Error loading data:", e)
            return None, None

    def train(self, epochs=10):
        X_train, y_train = self.load_data()
        if X_train is None or y_train is None:
            print(f"[Client {self.client_id}] Training aborted due to data issues.")
            return

        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.model(X_train)
            loss = self.criterion(output, y_train)
            if torch.isnan(loss):
                print(f"[Client {self.client_id}] Loss is NaN, stopping training!")
                break
            loss.backward()
            self.optimizer.step()
            print(f"[Client {self.client_id}] Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    def save_model(self):
        model_path = f"client_model_{self.client_id}.pth"
        torch.save(self.model.state_dict(), model_path)
        print(f"[Client {self.client_id}] Model saved as {model_path}")

    def predict(self, user_input):
        user_input_tensor = torch.tensor(user_input, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(user_input_tensor)
            prob = torch.sigmoid(output).item()
        return (1 if prob >= 0.5 else 0, prob)


# Initialize client
client_id = 3  # Change for each client
data_path = f"client_data_{client_id}.csv"
global_model_path = 'global_model.pth'
client = Client(client_id, data_path, global_model_path)


# Gradio Interface
def train_model(epochs):
    client.train(epochs)
    client.save_model()
    return "Model trained and saved successfully."

def predict_features(input_features):
    try:
        features = [float(x) for x in input_features.split(",")]
        pred_class, pred_prob = client.predict(features)
        return f"Prediction: {'Preterm Birth' if pred_class == 1 else 'Not Preterm Birth'} (Probability: {pred_prob:.4f})"
    except:
        return "Error: Please enter 14 numeric features separated by commas."


with gr.Blocks() as demo:
    gr.Markdown("## Preterm Birth Prediction")

    with gr.Tab("Train Model"):
        epochs_input = gr.Number(label="Number of Epochs", value=10)
        train_btn = gr.Button("Train Model")
        train_output = gr.Textbox(label="Training Output")
        train_btn.click(train_model, inputs=epochs_input, outputs=train_output)

    with gr.Tab("Predict"):
        input_box = gr.Textbox(label="Input Features (14 comma-separated values)", placeholder="e.g., 25, 22.5, 1,...")
        predict_btn = gr.Button("Predict")
        predict_output = gr.Textbox(label="Prediction Output")
        predict_btn.click(predict_features, inputs=input_box, outputs=predict_output)


if __name__ == "__main__":
    demo.launch(share=True)
