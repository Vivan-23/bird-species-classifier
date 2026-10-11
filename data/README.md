# Dataset Information: Bird Species Classification

## Primary Source
- **Hugging Face Repository**: [chriamue/bird-species-dataset](https://huggingface.co/datasets/chriamue/bird-species-dataset)
- **Original Source**: Kaggle 525 Bird Species Image Classification by Gerald Piosenka
- **Original Dataset Dimensions**: 525 bird species, colour RGB images (~224x224 pixels)

## Subset Selection Strategy
To maintain a high-quality, reproducible, and computationally efficient benchmark, exactly 10 bird species are selected deterministically using alphabetical ordering and a minimum sample threshold (>=120 usable images per species).

### Selected 10 Species (Exact Dataset Labels)
1. `ABBOTTS BABBLER`
2. `ABBOTTS BOOBY`
3. `ABYSSINIAN GROUND HORNBILL`
4. `AFRICAN CROWNED CRANE`
5. `AFRICAN EMERALD CUCKOO`
6. `AFRICAN FIREFINCH`
7. `AFRICAN OYSTER CATCHER`
8. `AFRICAN PIED HORNBILL`
9. `AFRICAN PYGMY GOOSE`
10. `ALBATROSS`

## Splitting Strategy
- **Target Proportions**: 70% Training, 15% Validation, 15% Testing per species.
- **Stratification**: Balanced per species using fixed random seed (`seed=42`).
- **Disjoint Partitioning**: No image appears in more than one split (zero intersection).

## Directory Layout
After running `python scripts/prepare_data.py`:
```
data/
├── metadata.json
└── processed/
    ├── train/
    │   ├── ABBOTTS BABBLER/
    │   └── ...
    ├── val/
    │   ├── ABBOTTS BABBLER/
    │   └── ...
    └── test/
        ├── ABBOTTS BABBLER/
        └── ...
```
All split counts, checksums, and mappings are stored in `data/metadata.json`.
