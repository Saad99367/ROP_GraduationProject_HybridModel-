import argparse
import torch
import json
import sys
import os

# Add the src directory to the python path so 'rop' can be imported without installation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from rop.inference.predictor import ROPPredictor

def main():
    parser = argparse.ArgumentParser(description="Predict ROP Stage")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = ROPPredictor(args.checkpoint, device=device)
    
    idx, name, probs = predictor.predict(args.image)
    
    print(f"Predicted Class: {idx} ({name})")
    print("Probabilities:")
    print(json.dumps(probs, indent=2))

if __name__ == "__main__":
    main()
