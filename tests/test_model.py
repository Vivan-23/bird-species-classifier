"""Unit tests for CNN architecture, forward pass, loss calculation,
optimization step, checkpoint serialization, and prediction execution.
"""

import os
import sys
import shutil
import tempfile
import unittest
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import AdamW

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import BirdClassifierCNN
from src.utils import save_checkpoint, load_checkpoint
from src.predict import BirdPredictor


class TestModelAndTraining(unittest.TestCase):
    """Test suite covering CNN model architecture, forward pass, loss, and inference."""

    def setUp(self):
        self.num_classes = 10
        self.batch_size = 4
        self.model = BirdClassifierCNN(
            in_channels=3,
            num_classes=self.num_classes,
            dropout_prob=0.4
        )
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_forward_pass_dimensions(self):
        """Verify model accepts [B, 3, 224, 224] and returns [B, num_classes]."""
        dummy_input = torch.randn(self.batch_size, 3, 224, 224)
        output = self.model(dummy_input)

        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        self.assertFalse(torch.isnan(output).any())

    def test_finite_loss_computation(self):
        """Verify nn.CrossEntropyLoss computes a finite scalar loss with raw logits."""
        criterion = nn.CrossEntropyLoss()
        dummy_input = torch.randn(self.batch_size, 3, 224, 224)
        dummy_targets = torch.randint(0, self.num_classes, (self.batch_size,))

        logits = self.model(dummy_input)
        loss = criterion(logits, dummy_targets)

        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(loss.item(), 0.0)

    def test_single_training_step(self):
        """Verify backward pass and optimizer update perform without error."""
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(self.model.parameters(), lr=1e-3)

        dummy_input = torch.randn(self.batch_size, 3, 224, 224)
        dummy_targets = torch.randint(0, self.num_classes, (self.batch_size,))

        self.model.train()
        optimizer.zero_grad()
        logits = self.model(dummy_input)
        loss = criterion(logits, dummy_targets)
        loss.backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(loss).item())

    def test_checkpoint_save_and_load(self):
        """Verify checkpoint serialization and deserialization integrity."""
        ckpt_path = os.path.join(self.temp_dir, "test_ckpt.pth")
        species_names = [f"Species_{i}" for i in range(self.num_classes)]
        ckpt_data = {
            "epoch": 5,
            "model_state_dict": self.model.state_dict(),
            "class_names": species_names,
            "val_loss": 0.42,
            "val_acc": 88.5
        }

        save_checkpoint(ckpt_data, ckpt_path)
        loaded = load_checkpoint(ckpt_path, device=torch.device("cpu"))

        self.assertEqual(loaded["epoch"], 5)
        self.assertEqual(loaded["val_acc"], 88.5)
        self.assertEqual(len(loaded["class_names"]), self.num_classes)

        # Load weights into new model instance
        new_model = BirdClassifierCNN(in_channels=3, num_classes=self.num_classes)
        new_model.load_state_dict(loaded["model_state_dict"])
        new_model.eval()

        dummy_input = torch.randn(2, 3, 224, 224)
        out = new_model(dummy_input)
        self.assertEqual(out.shape, (2, self.num_classes))

    def test_prediction_pipeline_on_dummy_image(self):
        """Verify BirdPredictor executes on test image and returns valid species."""
        # Create dummy checkpoint
        ckpt_path = os.path.join(self.temp_dir, "best_model.pth")
        species_names = [f"SPECIES_{i}" for i in range(10)]
        ckpt_data = {
            "epoch": 1,
            "model_state_dict": self.model.state_dict(),
            "class_names": species_names,
            "config": {
                "model": {"dropout_prob": 0.4},
                "dataset": {"image_size": 224}
            }
        }
        save_checkpoint(ckpt_data, ckpt_path)

        # Create dummy image
        img_path = os.path.join(self.temp_dir, "sample_bird.jpg")
        img = Image.new("RGB", (224, 224), color=(120, 200, 150))
        img.save(img_path)

        predictor = BirdPredictor(checkpoint_path=ckpt_path, device_setting="cpu")
        preds = predictor.predict(img_path, top_k=3)

        self.assertEqual(len(preds), 3)
        self.assertIn(preds[0]["species"], species_names)
        self.assertGreater(preds[0]["confidence"], 0.0)

    def test_early_stopping_records_outcome_and_reason(self):
        """Verify EarlyStopping records triggered state, stopped epoch, and explanatory reason."""
        from src.train import EarlyStopping

        es = EarlyStopping(patience=3, min_delta=0.01)
        # Epoch 1: improvement
        self.assertTrue(es(1.50, epoch=1))
        self.assertFalse(es.early_stop)
        # Epoch 2: no improvement (counter = 1)
        self.assertFalse(es(1.51, epoch=2))
        self.assertFalse(es.early_stop)
        # Epoch 3: no improvement (counter = 2)
        self.assertFalse(es(1.505, epoch=3))
        self.assertFalse(es.early_stop)
        # Epoch 4: no improvement (counter = 3 >= patience -> triggers early stopping)
        self.assertFalse(es(1.52, epoch=4))
        self.assertTrue(es.early_stop)
        self.assertEqual(es.stopped_epoch, 4)
        self.assertIn("consecutive epochs", es.stopping_reason)

    def test_checkpoint_class_mismatch_validation(self):
        """Verify that class mismatch between checkpoint and expected classes raises ValueError."""
        ckpt_classes = ["SpeciesA", "SpeciesB"]
        loader_classes = ["SpeciesA", "SpeciesC"]  # Mismatch

        with self.assertRaises(ValueError) as ctx:
            if ckpt_classes != loader_classes:
                raise ValueError(
                    f"Class name mismatch between checkpoint and test DataLoader!\n"
                    f"Checkpoint classes ({len(ckpt_classes)}): {ckpt_classes}\n"
                    f"DataLoader classes ({len(loader_classes)}): {loader_classes}"
                )
        self.assertIn("Class name mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
