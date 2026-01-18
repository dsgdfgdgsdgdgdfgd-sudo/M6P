# M6P Genomic Prediction

This repository contains data processing, training, and inference scripts for
multi-trait genomic prediction models (PIC-GD and related datasets).

## Setup

1) Create a Python environment and install dependencies:
```
pip install -r requirements.txt
```

2) Place data under `data/`:
```
data/
  PIC-GD/
  HZA-PMB/
```

## Run Inference

Run the enhanced model test/inference script (exports predictions only):
```
python scripts/test_enhanced_model.py
```

The script prints key paths and writes a CSV of predictions under:
```
experiments/enhanced_model_20251015_172800/
```

## Training

Train the enhanced multi-task model:
```
python scripts/enhanced_model_training.py
```


## Notes
- Update paths in scripts if you relocate the project directory.
