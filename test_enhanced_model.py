#!/usr/bin/env python3
"""
Run enhanced model inference on PIC-GD and export model predictions.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.multi_task_transformer import MultiTaskSNPTransformer
from scripts.enhanced_model_training import EnhancedModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_data():
    """Load PIC-GD, select top SNPs, split train/val/test, and build a test loader."""
    logger.info("Loading PIC-GD data and preparing splits...")

    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    data_path = 'data/PIC-GD/pic_gd_multi_task_preprocessed.csv'
    data_abs_path = os.path.abspath(data_path)
    print(f"[DATA] {data_abs_path}")
    with tqdm(total=1, desc="Loading PIC-GD data", unit="file") as pbar:
        data = pd.read_csv(data_path)
        pbar.update()

    target_phenotypes = ['t1', 't2', 't3', 't4', 't5']
    feature_cols = [col for col in data.columns if col not in target_phenotypes and col != 'Id']

    X_full = data[feature_cols].values
    y_full = data[target_phenotypes].values
    ids_full = data['Id'].values

    # Select the top 10k SNPs by average absolute correlation to targets
    correlations = []
    for i in tqdm(range(X_full.shape[1]), desc="loading SNPs", unit="snp"):
        feature_corr = []
        for j in range(y_full.shape[1]):
            corr = np.corrcoef(X_full[:, i], y_full[:, j])[0, 1]
            feature_corr.append(abs(corr) if not np.isnan(corr) else 0)
        correlations.append(np.mean(feature_corr))

    correlations = np.array(correlations)
    top_indices = np.argsort(correlations)[-10000:]
    selected_feature_cols = [feature_cols[i] for i in top_indices]
    X_selected = X_full[:, top_indices].astype(np.int64)

    # Split: 80% train, 10% val, 10% test
    n_samples = len(X_selected)
    n_train = int(n_samples * 0.8)
    n_val = int(n_samples * 0.1)

    rng = np.random.RandomState(random_seed)
    indices = rng.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train = X_selected[train_idx]
    X_val = X_selected[val_idx]
    X_test = X_selected[test_idx]
    y_train = y_full[train_idx]
    y_val = y_full[val_idx]
    y_test = y_full[test_idx]
    ids_train = ids_full[train_idx]
    ids_val = ids_full[val_idx]
    ids_test = ids_full[test_idx]

    # Label scaler (fit on full set to match prior script behavior)
    scaler = StandardScaler()
    scaler.fit(y_full)
    y_train_scaled = scaler.transform(y_train)
    y_val_scaled = scaler.transform(y_val)
    y_test_scaled = scaler.transform(y_test)

    # Test loader for enhanced model (uses scaled labels)
    X_test_tensor = torch.LongTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return {
        "test_loader": test_loader,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_test_scaled": y_test_scaled,
        "ids_train": ids_train,
        "ids_val": ids_val,
        "ids_test": ids_test,
        "scaler": scaler,
        "selected_feature_cols": selected_feature_cols,
        "top_indices": top_indices,
    }


def create_enhanced_model(config):
    """Create model from checkpoint config."""
    model = MultiTaskSNPTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_len=config.max_len,
        mode=config.mode,
        mlp_hidden_dims=config.mlp_hidden_dims,
    )
    return model


def evaluate_model_outputs(model, data_loader, device, task_names):
    """Return predictions and targets as numpy arrays (order matches task_names)."""
    model.eval()
    all_predictions = {task: [] for task in task_names}
    all_targets = []

    with torch.no_grad():
        for data, targets in tqdm(data_loader, desc="Running model inference", unit="batch"):
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            for task in task_names:
                all_predictions[task].append(outputs[task].cpu())
            all_targets.append(targets.cpu())

    all_predictions = {task: torch.cat(preds) for task, preds in all_predictions.items()}
    all_targets = torch.cat(all_targets)

    pred_matrix = np.vstack([all_predictions[task].numpy().flatten() for task in task_names]).T
    target_matrix = all_targets.numpy()
    return pred_matrix, target_matrix


def test_enhanced_model():
    print("=" * 100)
    print("M6P 增强模型推理")
    print("=" * 100)

    # 数据
    data_splits = load_test_data()
    test_loader = data_splits["test_loader"]
    ids_test = data_splits["ids_test"]
    scaler = data_splits["scaler"]

    # 模型检查点
    model_path = "experiments/enhanced_model_20251015_172800/best_model.pth"
    model_abs_path = os.path.abspath(model_path)
    print(f"[MODEL] {model_abs_path}")

    if not os.path.exists(model_path):
        logger.error(f"Enhanced model not found: {model_path}")
        return None

    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']

    logger.info("Loading enhanced model checkpoint...")
    model = create_enhanced_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    logger.info("Model checkpoint loaded successfully")

    task_names = ['t1', 't2', 't3', 't4', 't5']

    # 推理
    print(f"[INFER] Running inference on {len(test_loader.dataset)} samples...")
    pred_scaled, _ = evaluate_model_outputs(model, test_loader, device, task_names)
    pred_enhanced = scaler.inverse_transform(pred_scaled)

    # 仅导出模型预测值
    output_dir = Path(model_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(len(ids_test)):
        row = {'Id': ids_test[i]}
        for j, task in enumerate(task_names):
            row[f"{task}_pred_enhanced"] = float(pred_enhanced[i, j])
        rows.append(row)
    pred_df = pd.DataFrame(rows)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_csv_path = output_dir / f"{ts_str}_M6P_predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"[OUTPUT] Predictions saved: {pred_csv_path.resolve()}")
    logger.info("Inference finished.")
    return pred_csv_path


def main():
    pred_path = test_enhanced_model()
    if pred_path:
        print(f"[DONE] Prediction file ready: {Path(pred_path).resolve()}")


if __name__ == "__main__":
    main()
