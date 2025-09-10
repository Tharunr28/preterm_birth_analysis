import os
import time
import torch
import torch.nn as nn
from threading import Thread
import gradio as gr
from model import SimpleNN  # Make sure SimpleNN is defined in model.py


# Initialize the global model
input_size = 14
global_model = SimpleNN(input_size)


# Function to safely average weights, ignoring NaNs
def update_global_model(local_weights_list):
    global_dict = global_model.state_dict()

    for key in global_dict.keys():
        key_weights = []
        for local_weights in local_weights_list:
            if key in local_weights:
                tensor = local_weights[key]
                # Replace NaNs with zeros
                tensor = torch.nan_to_num(tensor, nan=0.0)
                key_weights.append(tensor)
            else:
                print(f"Warning: Client missing key {key}")

        if key_weights:
            # Stack and average
            global_dict[key] = torch.stack(key_weights, 0).mean(0)
        else:
            print(f"Skipping key {key} as no clients provided weights.")

    global_model.load_state_dict(global_dict)


# Load client weights safely
def fetch_local_weights():
    local_weights_list = []
    for client_id in range(1, 6):  # Assuming 5 clients
        try:
            local_weights = torch.load(f'client_model_{client_id}.pth')
            # Replace NaNs in each tensor
            for k in local_weights:
                local_weights[k] = torch.nan_to_num(local_weights[k], nan=0.0)
            print(f"[Client {client_id}] Weights loaded.")
            local_weights_list.append(local_weights)
        except FileNotFoundError:
            print(f"[Client {client_id}] Model file not found.")
        except Exception as e:
            print(f"[Client {client_id}] Error loading weights:", e)
    return local_weights_list


# Periodically update the global model
def periodic_update():
    while True:
        print("[Server] Fetching local weights...")
        local_weights_list = fetch_local_weights()
        if local_weights_list:
            update_global_model(local_weights_list)
            torch.save(global_model.state_dict(), 'global_model.pth')
            print("[Server] Global model updated and saved.")
        else:
            print("[Server] No local weights available.")
        time.sleep(10)  # every 10 seconds


# Manual update via Gradio
def gradio_update():
    local_weights_list = fetch_local_weights()
    if local_weights_list:
        update_global_model(local_weights_list)
        torch.save(global_model.state_dict(), 'global_model.pth')
        return "Global model updated successfully."
    else:
        return "No local weights available to update."


# Prediction function
def predict(input_features):
    try:
        input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = global_model(input_tensor)
            prediction = torch.sigmoid(output).item()
        return "Preterm Birth" if prediction >= 0.5 else "Not Preterm Birth", prediction
    except Exception as e:
        return f"Error in prediction: {e}", 0.0


# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Preterm Birth Prediction Model")

    with gr.Tab("Update Model"):
        update_button = gr.Button("Update Global Model")
        update_output = gr.Textbox(label="Update Status")
        update_button.click(gradio_update, outputs=update_output)

    with gr.Tab("Predict"):
        input_features = gr.Textbox(label="Input Features (comma-separated, 14 values)")
        predict_button = gr.Button("Make Prediction")
        prediction_output = gr.Textbox(label="Prediction Output")
        predict_button.click(
            lambda x: predict([float(i) if i.strip() else 0.0 for i in x.split(",")]),
            inputs=input_features,
            outputs=prediction_output
        )


# Run the periodic update thread and Gradio app
if __name__ == "__main__":
    update_thread = Thread(target=periodic_update, daemon=True)
    update_thread.start()
    demo.launch(server_port=7861, share=True)
