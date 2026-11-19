# -*- coding: utf-8 -*-
"""CPBL 2026 逐打席 MLP 訓練（四分類、H2H，Dense 64→32→16→4）。

執行 python train_cpbl_mlp_h2h.py 即會自動呼叫 preprocess_cpbl_h2h.py。
固定跑完 40 個 Epoch，不啟用 EarlyStopping；儲存驗證損失最低的模型。
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from preprocess_cpbl_h2h import PA_FILE, prepare_data

ROOT = Path(__file__).resolve().parent
SEED = 35
EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# 此處沿用你原本設定的四分類權重，可再依驗證集成績調整。
# 0=出局、1=一安、2=長打、3=保送（含觸身球）
CLASS_WEIGHT = {
                 0: 0.5,
                 1: 1.9, 
                 2: 2.6, 
                 3: 2.1
                }

MODEL_FILE = ROOT / "cpbl_mlp_h2h_4class.keras"
PREPROCESSOR_FILE = ROOT / "cpbl_mlp_h2h_4class_preprocessor.joblib"




def create_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    tf.keras.utils.set_random_seed(SEED)

    # 前處理在獨立檔案，且訓練時自動呼叫。
    data = prepare_data(PA_FILE)
    X_train, X_valid, X_test = data["X_train"], data["X_valid"], data["X_test"]
    y_train, y_valid, y_test = data["y_train"], data["y_valid"], data["y_test"]
    labels = data["labels"]
    n_classes = len(labels)




    model = create_model(X_train.shape[1])
    model.summary()

    # 沒有 EarlyStopping：40 個 Epoch 會全部執行。
    # 但只保留 val_loss 最低時的模型供最終測試與後續預測。
    save_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_FILE),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=0,
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=CLASS_WEIGHT,
        callbacks=[save_best],
        verbose=2,
    )



    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    best_model = tf.keras.models.load_model(MODEL_FILE)
    probs = best_model.predict(X_test, verbose=0)
    pred = probs.argmax(axis=1)
    baseline_label = int(np.bincount(y_train, minlength=n_classes).argmax())

    report = {
        "original_rows": data["original_rows"],
        "valid_rows": data["valid_rows"],
        "train_rows": len(data["train"]),
        "validation_rows": len(data["valid"]),
        "test_rows": len(data["test"]),
        "input_dim": int(X_train.shape[1]),
        "architecture": [64, 32, 16, 4],
        "epochs_completed": len(history.history["loss"]),
        "best_epoch": best_epoch,
        "best_val_loss": float(min(history.history["val_loss"])),
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "class_weight": {labels[i]: float(CLASS_WEIGHT[i]) for i in range(n_classes)},
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_test, probs, labels=list(range(n_classes)))),
        "baseline_accuracy": float(np.mean(y_test == baseline_label)),
        "classes": labels,
        "actual_class_counts": {
            name: int((y_test == i).sum()) for i, name in enumerate(labels)
        },
        "predicted_class_counts": {
            name: int((pred == i).sum()) for i, name in enumerate(labels)
        },
        "ignored_errors": data["ignored_errors"],
        "ignored_sacrifices_bunts": data["ignored_sacrifices_bunts"],
        "ignored_other": data["ignored_other"],
        "h2h_note": "H2H 只使用較早比賽日期的四分類打席；同日打席不計入。",
        "data_note": "不使用 2026 全年球員統計回填歷史特徵，降低未來資料洩漏風險。",
    }

    joblib.dump({
        "preprocessor": data["prep"],
        "categorical_columns": data["cats"],
        "numeric_columns": data["nums"],
        "labels": labels,
    }, PREPROCESSOR_FILE)

    output_columns = [
        "比賽日期", "打者", "投手", "打擊結果", "標籤",
        "h2h_prior_pa_count", "h2h_out_rate", "h2h_single_rate",
        "h2h_xbh_rate", "h2h_walk_rate",
    ]
    out = data["test"][output_columns].copy()
    out["預測結果"] = [labels[int(i)] for i in pred]
    for i, name in enumerate(labels):
        out["機率_" + name] = probs[:, i]

    print("\n========== 測試結果（驗證損失最低的模型） ==========")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\n分類報告：")
    print(classification_report(
        y_test, pred, labels=list(range(n_classes)), target_names=labels, zero_division=0
    ))
    print("混淆矩陣（列=真實，欄=預測；順序：出局、一安、長打、保送）：")
    print(confusion_matrix(y_test, pred, labels=list(range(n_classes))))

    print("\n========== 全部 Epoch 訓練紀錄 ==========")
    for i in range(len(history.history["loss"])):
        print(
            f"Epoch {i + 1:02d} | "
            f"Accuracy: {history.history['accuracy'][i]:.4f} | "
            f"Loss: {history.history['loss'][i]:.4f} | "
            f"Val Accuracy: {history.history['val_accuracy'][i]:.4f} | "
            f"Val Loss: {history.history['val_loss'][i]:.4f}"
        )



if __name__ == "__main__":
    main()
