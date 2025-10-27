#!/usr/bin/env python3
"""
CS 599 — Lab 2: Measuring Catastrophic Forgetting on Permuted-MNIST
TensorFlow 2.x + Keras implementation

Features
- Permuted MNIST with T tasks (default 10)
- MLP depths: 2, 3, 4 hidden layers (each 256 units)
- Loss setting: NLL (+ optional L1/L2/L1+L2 kernel regularization)
- Optimizers: SGD, Adam, RMSprop
- Dropout up to 0.5
- Training schedule: 50 epochs for Task 1, 20 for remaining tasks (total 230 epochs)
- Computes R matrix (task-by-task accuracy), ACC, BWT, TBWT, CBWT
- Saves artifacts: R.npy, R.csv, metrics.json, plots

CLI examples
- python3 forgetting_mlp.py --depth 3 --loss nll --optimizer adam --dropout 0.3
- python3 forgetting_mlp.py --depth all --loss nll --optimizer sgd --dropout 0.5 --outdir outputs/sgd_nodrop

Author: Generated for CS 599 Lab 2 (Pavan)
"""
import argparse
import os
import json
import math
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------
# Utilities
# ------------------------

def set_global_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # normalize to [0,1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)


def make_permutation(seed: int, size: int = 28 * 28) -> np.ndarray:
    rng = np.random.RandomState(seed)
    perm = np.arange(size)
    rng.shuffle(perm)
    return perm


def apply_permutation(x: np.ndarray, perm: np.ndarray) -> np.ndarray:
    # x: (N, 28, 28) -> (N, 28*28) permuted -> back to (N, 28, 28)
    n = x.shape[0]
    flat = x.reshape(n, -1)[:, perm]
    return flat.reshape(n, 28, 28)


def build_mlp(input_shape=(28, 28), depth=3, units=256, num_classes=10,
              dropout=0.5, reg_type="none", reg_scale=1e-4) -> keras.Model:
    """
    reg_type in {"none", "l1", "l2", "l1l2"}
    """
    if reg_type == "l1":
        kr = regularizers.l1(reg_scale)
    elif reg_type == "l2":
        kr = regularizers.l2(reg_scale)
    elif reg_type == "l1l2":
        kr = regularizers.l1_l2(l1=reg_scale, l2=reg_scale)
    else:
        kr = None

    inputs = keras.Input(shape=input_shape)
    x = layers.Flatten()(inputs)

    for _ in range(depth):
        x = layers.Dense(units, activation="relu", kernel_regularizer=kr)(x)
        if dropout and dropout > 0.0:
            x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def get_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=lr)
    raise ValueError(f"Unknown optimizer: {name}")


def compute_metrics(R: np.ndarray) -> Dict[str, float]:
    """
    ACC = (1/T) * sum_i R[T-1, i]
    BWT = (1/(T-1)) * sum_{i=0}^{T-2} (R[T-1, i] - R[i, i])
    TBWT (Total BWT): mean over all pairs j>i of (R[j, i] - R[i, i])
    CBWT (Cross-BWT): mean over offsets k>=1 of average (R[i+k, i] - R[i, i])
    """
    T = R.shape[0]
    ACC = R[T-1].mean()
    BWT = np.mean([R[T-1, i] - R[i, i] for i in range(T-1)])

    # TBWT
    diffs = []
    for j in range(1, T):
        for i in range(j):
            diffs.append(R[j, i] - R[i, i])
    TBWT = float(np.mean(diffs)) if diffs else 0.0

    # CBWT: group by offset k = j-i
    cbwt_vals = []
    for k in range(1, T):
        vals = []
        for i in range(0, T-k):
            j = i + k
            vals.append(R[j, i] - R[i, i])
        cbwt_vals.append(np.mean(vals))
    CBWT = float(np.mean(cbwt_vals)) if cbwt_vals else 0.0

    return {"ACC": float(ACC), "BWT": float(BWT), "TBWT": float(TBWT), "CBWT": float(CBWT)}


def plot_last_row(R: np.ndarray, savepath: str, title: str):
    T = R.shape[0]
    plt.figure(figsize=(7,4.5))
    plt.plot(np.arange(1, T+1), R[T-1], marker="o")
    plt.xlabel("Task index")
    plt.ylabel("Accuracy after finishing all tasks")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(savepath, dpi=160)
    plt.close()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ------------------------
# Main experiment
# ------------------------

def run_one(depth: int, loss: str, optimizer: str, dropout: float, args):
    assert 0.0 <= dropout <= 0.5, "Dropout must be in [0.0, 0.5] per assignment."
    set_global_seed(args.seed)

    (x_train, y_train), (x_test, y_test) = load_mnist()

    # prepare task permutations and permuted test sets upfront
    perms = [make_permutation(args.seed + t, 28*28) for t in range(args.tasks)]
    test_sets = []
    for t in range(args.tasks):
        x_t = apply_permutation(x_test, perms[t])
        test_sets.append((x_t, y_test))

    # map loss string to regularizer setting
    loss_key = loss.lower()
    if loss_key == "nll":
        reg_type = "none"
    elif loss_key == "l1":
        reg_type = "l1"
    elif loss_key == "l2":
        reg_type = "l2"
    elif loss_key in ("l1+l2", "l1l2"):
        reg_type = "l1l2"
    else:
        raise ValueError("loss must be one of: nll, l1, l2, l1+l2")

    model = build_mlp(depth=depth, dropout=dropout, reg_type=reg_type, reg_scale=args.reg_scale)

    opt = get_optimizer(optimizer, args.lr)
    model.compile(optimizer=opt,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=["accuracy"])

    # R matrix (T x T)
    R = np.zeros((args.tasks, args.tasks), dtype=np.float32)

    # train sequentially across tasks
    for t in range(args.tasks):
        x_tr = apply_permutation(x_train, perms[t])
        # epochs per schedule
        epochs = args.epochs_first if t == 0 else args.epochs_rest

        model.fit(x_tr, y_train, batch_size=args.batch, epochs=epochs, verbose=args.verbosity,
                  validation_split=0.1, shuffle=True)

        # evaluate on all seen tasks so far (and beyond, for a full R row t)
        for i in range(args.tasks):
            x_ev, y_ev = test_sets[i]
            _, acc = model.evaluate(x_ev, y_ev, batch_size=512, verbose=0)
            R[t, i] = acc

    # metrics
    metrics = compute_metrics(R)

    # save artifacts
    tag = f"d{depth}_{loss_key}_{optimizer}_do{dropout:.1f}_seed{args.seed}"
    outdir = os.path.join(args.outdir, tag)
    ensure_dir(outdir)

    np.save(os.path.join(outdir, "R.npy"), R)
    np.savetxt(os.path.join(outdir, "R.csv"), R, delimiter=",", fmt="%.6f")
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    title = f"Last-row accuracy | depth={depth}, loss={loss_key}, opt={optimizer}, do={dropout:.1f}"
    plot_last_row(R, os.path.join(outdir, "last_row.png"), title)

    # simple text summary
    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write(title + "\n")
        f.write("ACC={ACC:.4f}  BWT={BWT:.4f}  TBWT={TBWT:.4f}  CBWT={CBWT:.4f}\n".format(**metrics))
        f.write("\nR (accuracy by task after each training step):\n")
        f.write(str(R))

    print(f"\nSaved results to: {outdir}")
    print(json.dumps(metrics, indent=2))


def main():
    p = argparse.ArgumentParser(description="CS 599 Lab 2 — Catastrophic Forgetting on Permuted-MNIST")
    p.add_argument("--tasks", type=int, default=10, help="Number of tasks (default: 10)")
    p.add_argument("--epochs_first", type=int, default=50, help="Epochs for Task 1 (default: 50)")
    p.add_argument("--epochs_rest", type=int, default=20, help="Epochs for Tasks 2..T (default: 20)")
    p.add_argument("--batch", type=int, default=64, help="Batch size (default: 64)")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--seed", type=int, default=5695, help="Global seed (use your unique seed)")
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd", "rmsprop"])
    p.add_argument("--loss", type=str, default="nll", choices=["nll", "l1", "l2", "l1+l2", "l1l2"])
    p.add_argument("--dropout", type=float, default=0.5, help="Dropout in [0, 0.5]")
    p.add_argument("--depth", type=str, default="3",
                   help="Hidden depth: 2,3,4, or 'all' to sweep all three")
    p.add_argument("--outdir", type=str, default="outputs", help="Directory to save results")
    p.add_argument("--reg_scale", type=float, default=1e-4, help="Regularization scale for L1/L2/L1+L2")
    p.add_argument("--verbosity", type=int, default=2, help="Keras fit verbosity (0,1,2)")
    args = p.parse_args()

    ensure_dir(args.outdir)

    depths: List[int]
    if args.depth == "all":
        depths = [2,3,4]
    else:
        d = int(args.depth)
        assert d in (2,3,4), "--depth must be 2, 3, 4, or 'all'"
        depths = [d]

    for d in depths:
        run_one(depth=d, loss=args.loss, optimizer=args.optimizer, dropout=args.dropout, args=args)


if __name__ == "__main__":
    main()
