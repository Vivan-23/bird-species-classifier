"""CLI Inference Script for Bird Species Classification.

Usage:
    python src/predict.py --image path/to/bird.jpg
    python src/predict.py --image path/to/bird.jpg --top_k 3 --checkpoint models/best_model.pth
"""

import os
import sys
import argparse
from PIL import Image
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import load_checkpoint, get_device
from src.dataset import get_eval_transforms
from src.model import BirdClassifierCNN


class BirdPredictor:
    """Inference engine for Bird Species Classification."""

    def __init__(self, checkpoint_path: str = "models/best_model.pth", device_setting: str = "auto"):
        self.device = get_device(device_setting)
        self.checkpoint_path = os.path.abspath(checkpoint_path)

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at '{self.checkpoint_path}'. "
                "Please train the model first by running: python src/train.py"
            )

        checkpoint = load_checkpoint(self.checkpoint_path, device=self.device)
        self.class_names = checkpoint["class_names"]
        self.class_to_idx = checkpoint.get("class_to_idx", {name: i for i, name in enumerate(self.class_names)})

        # Reconstruct CNN
        config = checkpoint.get("config", {})
        dropout_prob = config.get("model", {}).get("dropout_prob", 0.4)
        image_size = config.get("dataset", {}).get("image_size", 224)

        self.model = BirdClassifierCNN(
            in_channels=3,
            num_classes=len(self.class_names),
            dropout_prob=dropout_prob
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.transform = get_eval_transforms(image_size=image_size)

    @torch.no_grad()
    def predict(self, image_input: str | Image.Image, top_k: int = 3) -> list[dict]:
        """Classify a bird image from either a file path or an in-memory PIL Image.

        Args:
            image_input: File path string or PIL Image instance.
            top_k: Number of top-ranked predictions to return (sanitized between 1 and num_classes).

        Returns:
            List of dicts: [{'rank': 1, 'species': name, 'probability': float, 'confidence': float, 'percentage': str}, ...]
        """
        # Top-k boundary sanitization
        try:
            top_k = max(1, min(int(top_k), len(self.class_names)))
        except (ValueError, TypeError):
            top_k = min(3, len(self.class_names))

        # Handle PIL Image or filepath input
        if isinstance(image_input, Image.Image):
            rgb_img = image_input.convert("RGB")
        elif isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Provided image path does not exist: {image_input}")
            try:
                with Image.open(image_input) as raw_img:
                    rgb_img = raw_img.convert("RGB")
            except Exception as e:
                raise ValueError(f"Unable to read or parse image file '{image_input}': {e}")
        else:
            raise TypeError(
                f"Unsupported input type '{type(image_input).__name__}'. "
                "Expected file path string or PIL.Image.Image instance."
            )

        # Preprocess and batchify
        tensor = self.transform(rgb_img).unsqueeze(0).to(self.device)

        # Forward pass (eval mode ensures Dropout is inactive)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

        # Retrieve top-k
        top_probs, top_indices = torch.topk(probs, k=top_k)

        results = []
        for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), start=1):
            species_name = self.class_names[idx.item()]
            prob_val = float(prob.item())
            results.append({
                "rank": rank,
                "species": species_name,
                "probability": round(prob_val, 4),
                "confidence": round(prob_val, 4),
                "percentage": f"{prob_val * 100:.2f}%"
            })
        return results


def main():
    parser = argparse.ArgumentParser(description="Predict bird species from an image file.")
    parser.add_argument("--image", type=str, required=True, help="Path to bird image.")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pth", help="Checkpoint file path.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions to display.")
    parser.add_argument("--device", type=str, default="auto", help="Inference device ('auto', 'cpu', 'cuda').")
    args = parser.parse_args()

    try:
        predictor = BirdPredictor(checkpoint_path=args.checkpoint, device_setting=args.device)
        predictions = predictor.predict(args.image, top_k=args.top_k)

        print("\n" + "=" * 55)
        print(f"IMAGE: {args.image}")
        print("=" * 55)
        top_result = predictions[0]
        print(f"PREDICTED SPECIES: {top_result['species']}")
        print(f"CONFIDENCE:        {top_result['percentage']}")
        print("-" * 55)
        print("TOP PREDICTIONS:")
        for res in predictions:
            print(f"  #{res['rank']} {res['species']:<28} {res['percentage']}")
        print("=" * 55 + "\n")

    except Exception as e:
        print(f"\n[Error] Prediction failed: {e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
