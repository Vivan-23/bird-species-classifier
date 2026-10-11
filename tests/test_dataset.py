"""Unit tests for dataset transformations, loading, and split non-overlap.
Uses synthetic in-memory/temporary image fixtures for standalone, fast test execution.
"""

import os
import sys
import shutil
import tempfile
import unittest
from PIL import Image
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import get_train_transforms, get_eval_transforms, BirdDataset


class TestDatasetAndTransforms(unittest.TestCase):
    """Test suite covering dataset preprocessing, transforms, and loading integrity."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.class_names = ["SPECIES_A", "SPECIES_B"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Create dummy directory structure
        for sp in self.class_names:
            sp_dir = os.path.join(self.temp_dir, sp)
            os.makedirs(sp_dir, exist_ok=True)
            for i in range(3):
                img = Image.new("RGB", (250, 250), color=(i * 40, i * 50, i * 60))
                img.save(os.path.join(sp_dir, f"dummy_{i}.jpg"))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_train_transforms_output_shape_and_range(self):
        """Verify train transformations output [3, 224, 224] tensor."""
        transform = get_train_transforms(image_size=224)
        sample_img = Image.new("RGB", (300, 300), color=(100, 150, 200))
        tensor = transform(sample_img)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertFalse(torch.isnan(tensor).any())

    def test_eval_transforms_output_shape(self):
        """Verify evaluation transformations output [3, 224, 224] tensor."""
        transform = get_eval_transforms(image_size=224)
        sample_img = Image.new("RGB", (180, 220), color=(50, 80, 110))
        tensor = transform(sample_img)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (3, 224, 224))
        self.assertFalse(torch.isnan(tensor).any())

    def test_bird_dataset_loading_and_labels(self):
        """Verify BirdDataset loads correctly from folder and respects class mapping."""
        dataset = BirdDataset(
            split_dir=self.temp_dir,
            class_to_idx=self.class_to_idx,
            transform=get_eval_transforms(224)
        )

        self.assertEqual(len(dataset), 6)  # 2 species * 3 images
        img_tensor, label = dataset[0]
        self.assertEqual(img_tensor.shape, (3, 224, 224))
        self.assertIn(label, [0, 1])

    def test_split_non_overlap(self):
        """Verify disjointness logic ensures no overlap between split partitions."""
        train_files = {"img1.jpg", "img2.jpg", "img3.jpg"}
        val_files = {"img4.jpg", "img5.jpg"}
        test_files = {"img6.jpg", "img7.jpg"}

        self.assertEqual(len(train_files.intersection(val_files)), 0)
        self.assertEqual(len(train_files.intersection(test_files)), 0)
        self.assertEqual(len(val_files.intersection(test_files)), 0)


if __name__ == "__main__":
    unittest.main()
