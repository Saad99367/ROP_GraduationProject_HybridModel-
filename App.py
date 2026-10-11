import gradio as gr
import torch
import os
import sys

# Add the src folder to path so we can import the 'rop' package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rop.inference.predictor import ROPPredictor

# Define model path
MODEL_PATH = "model/best_rop_model.pth"

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    predictor = ROPPredictor(checkpoint_path=MODEL_PATH, device=device)
    print(f"Model loaded successfully on {device}")
except Exception as e:
    print(f"Failed to load model from {MODEL_PATH}: {e}")
    predictor = None

def predict_rop(image_path):
    if predictor is None:
        return {"Error: Model not loaded properly.": 1.0}
    if image_path is None:
        return None
        
    try:
        class_idx, class_name, prob_dict = predictor.predict(image_path)
        return prob_dict
    except Exception as e:
        return {f"Error predicting: {str(e)}": 1.0}

# Create Gradio interface
iface = gr.Interface(
    fn=predict_rop,
    inputs=gr.Image(type="filepath", label="Upload Retinal Image"),
    outputs=gr.Label(num_top_classes=5, label="Prediction Probabilities"),
    title="Retinopathy of Prematurity (ROP) Classification",
    description="Upload a retinal image to predict the ROP stage using the Hybrid Fuzzy Vision Transformer model.",
    examples=[
        # Add examples here if you have standard paths
    ]
)

if __name__ == "__main__":
    iface.launch()
