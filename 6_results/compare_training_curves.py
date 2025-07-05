#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_training_curves.py

Plot Training and Validation Loss curves (MSE and MAE) for multiple traffic-forecasting models.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def load_metrics(model_name, model_dir):
    """
    Load metrics.csv from the given directory.  
    Expects columns: epoch, train_mse, val_mse, train_mae, val_mae.
    """
    path = os.path.join(model_dir, 'metrics.csv')
    if not os.path.isfile(path):
        raise FileNotFoundError(f"metrics.csv not found for {model_name} in {model_dir}")
    df = pd.read_csv(path)
    required = {'epoch', 'train_mse', 'val_mse', 'train_mae', 'val_mae'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{model_name} metrics.csv is missing columns: {missing}")
    return df

def plot_curves(metrics_dict, loss_type='mse'):
    """
    Plot train vs val curves for given loss_type ('mse' or 'mae').
    """
    plt.figure(figsize=(8, 5))
    for name, df in metrics_dict.items():
        train_col = f'train_{loss_type}'
        val_col   = f'val_{loss_type}'
        plt.plot(df['epoch'], df[train_col], linestyle='--', marker='o', label=f'{name} Train')
        plt.plot(df['epoch'], df[val_col],   linestyle='-',  marker='x', label=f'{name} Val')
    plt.xlabel('Epoch')
    plt.ylabel(f'{loss_type.upper()} Loss')
    plt.title(f'Training vs Validation {loss_type.upper()}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'comparison_{loss_type}.png', dpi=150)
    plt.show()

def main():
    # 1) Define your model directories here
    model_dirs = {
        'LSTM':        'ck_lstm',
        'ST-GCN':      'ck_stgcn',
        'PooledRes':   'ck_pooled'
    }

    # 2) Load all metrics
    metrics = {}
    for name, d in model_dirs.items():
        try:
            metrics[name] = load_metrics(name, d)
        except Exception as e:
            print(f"Skipping {name}: {e}")

    if not metrics:
        print("No metrics loaded. Exiting.")
        return

    # 3) Plot MSE curves
    plot_curves(metrics, loss_type='mse')

    # 4) (Optional) Plot MAE curves
    plot_curves(metrics, loss_type='mae')

if __name__ == '__main__':
    main()