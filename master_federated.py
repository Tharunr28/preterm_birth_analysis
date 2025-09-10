import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threading import Thread
import gradio as gr

# ----------------------------
# Import the model
# ----------------------------
from model import SimpleNN

# ----------------------------
# Client Class
# ----------------------------
class Client:
    def __init__(self, client_id, data_path, global_model):
        self.client_id = client_id
        self.data_path = data_path
        self.model = SimpleNN(input_size=14, hidden_size=64)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.global_model = global_model
        self.load_global_model()

    def load_global_model(self):
        try:
            self.model.load_state_dict(self.global_model.state_dict())
            self.model.eval()
            print(f"[Client {self.client_id}] Global model loaded.")
        except:
            print(f"[Client {self.client_id}] No global model, starting from scratch.")

    def load_data(self):
        try:
            data = pd.read_csv(self.data_path)
            data = data.apply(pd.to_numeric, errors='coerce')
            data.fillna(data.mean(), inplace=True)
            target = 'Preterm_Birth'
            X = data.drop(columns=[target]).values
            y = data[target].values
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X_train = torch.tensor(X, dtype=torch.float32)
            y_train = torch.tensor(y, dtype=torch.float32).view(-1,1)
            return X_train, y_train
        except Exception as e:
            print(f"[Client {self.client_id}] Data loading error:", e)
            return None, None

    def train(self, epochs=10):
        X_train, y_train = self.load_data()
        if X_train is None: return
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.model(X_train)
            loss = self.criterion(output, y_train)
            loss.backward()
            self.optimizer.step()
            print(f"[Client {self.client_id}] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    def save_model(self):
        path = f"client_model_{self.client_id}.pth"
        torch.save(self.model.state_dict(), path)
        print(f"[Client {self.client_id}] Model saved as {path}")

# ----------------------------
# Global Model Setup
# ----------------------------
global_model = SimpleNN(input_size=14)

def fetch_client_weights(num_clients=5):
    weights = []
    for cid in range(1, num_clients+1):
        try:
            w = torch.load(f'client_model_{cid}.pth')
            weights.append(w)
        except:
            print(f"[Server] Client {cid} model not found.")
    return weights

def aggregate_global_model():
    client_weights = fetch_client_weights()
    if not client_weights: return
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        key_weights = [w[key] for w in client_weights if key in w]
        if key_weights:
            global_dict[key] = torch.stack(key_weights,0).mean(0)
    global_model.load_state_dict(global_dict)
    torch.save(global_model.state_dict(), 'global_model.pth')
    print("[Server] Global model updated and saved.")

# ----------------------------
# Periodic automatic global model update
# ----------------------------
def periodic_update(interval=10):
    while True:
        aggregate_global_model()
        time.sleep(interval)

# ----------------------------
# Gradio functions
# ----------------------------
def gradio_update():
    aggregate_global_model()
    return "Global model updated manually."

def predict(input_features):
    with torch.no_grad():
        tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)
        output = global_model(tensor)
        prob = torch.sigmoid(output).item()
        return "Preterm Birth" if prob>=0.5 else "Not Preterm Birth", prob

# ----------------------------
# Train all clients automatically
# ----------------------------
def train_all_clients(epochs=10, num_clients=5):
    for cid in range(1, num_clients+1):
        client = Client(cid, f"client_data_{cid}.csv", global_model)
        client.train(epochs=epochs)
        client.save_model()
    print("[Server] All clients trained.")

# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    # Step 1: Train all clients
    train_all_clients(epochs=10)

    # Step 2: Aggregate into global model
    aggregate_global_model()

    # Step 3: Start periodic update in background
    update_thread = Thread(target=periodic_update, args=(10,))
    update_thread.start()

    # Step 4: Launch Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("## Federated Preterm Birth Prediction Server")
        with gr.Tab("Update Model"):
            btn = gr.Button("Update Global Model")
            out = gr.Textbox(label="Status")
            btn.click(gradio_update, outputs=out)
        with gr.Tab("Predict"):
            features = gr.Textbox(label="Input Features (comma-separated)")
            pred_btn = gr.Button("Make Prediction")
            pred_out = gr.Textbox(label="Prediction Output")
            pred_btn.click(lambda x: predict([float(i) for i in x.split(",")]), inputs=features, outputs=pred_out)

    demo.launch(share=True, server_port=7861)
