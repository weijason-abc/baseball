from pathlib import Path
import json
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, f1_score
from sklearn.utils.class_weight import compute_class_weight
from preprocess_cpbl_h2h import prepare_data
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
PA_FILE = ROOT / 'rebas_CPBL_2026_六隊逐打席_修正版.xlsx'
PROFILE_FILE = ROOT / '2026球員數據(含投手球種).xlsx'
SEED = 35
EPOCHS = 40
BATCH_SIZE = 32


def main():
    tf.keras.utils.set_random_seed(SEED)
    data = prepare_data(PA_FILE, PROFILE_FILE)
    train = data['train']
    valid = data['valid']
    test = data['test']
    X_train = data['X_train']
    X_valid = data['X_valid']
    X_test = data['X_test']
    y_train = data['y_train']
    y_valid = data['y_valid']
    y_test = data['y_test']
    prep = data['prep']
    cats = data['cats']
    nums = data['nums']
    labels = data['labels']
    n_classes = len(labels)
    classes = np.unique(y_train)
    
   
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    class_weight = {
                0: 0.7,  # 出局
                1: 1.2,  # 一安
                2: 2.2,  # 長打
                3: 1.6}   # 保送
    history=model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        class_weight=class_weight,
        verbose=2
    )
    

    probs = model.predict(X_test, verbose=0)
    pred = probs.argmax(axis=1)
    baseline_label = np.bincount(y_train, minlength=n_classes).argmax()
    auc = roc_auc_score(
        y_test,
        probs,
        multi_class='ovr',
        average='macro',
        labels=[0, 1, 2, 3]
    )

    report = {
        'train_rows': len(train),
        'validation_rows': len(valid),
        'test_rows': len(test),
        'ignored_errors': data['ignored_errors'],
        'ignored_sacrifices_bunts': data['ignored_sacrifices_bunts'],
        'ignored_other': data['ignored_other'],
        'accuracy': float(accuracy_score(y_test, pred)),
        'macro_f1': float(f1_score(y_test, pred, average='macro', zero_division=0)),
        'log_loss': float(log_loss(y_test, probs, labels=list(range(n_classes)))),
        'baseline_accuracy': float(np.mean(y_test == baseline_label)),
        'classes': labels,
        'class_weight': {labels[c]: w for c, w in class_weight.items()},
        'predicted_class_counts': {
            label: int((pred == i).sum()) for i, label in enumerate(labels)
        },
        'h2h_feature_note': 'H2H 僅使用比賽日期之前符合四分類的打席；同日打席不計入。',
        'warning': '全年球員統計及投手球種快照未用於歷史訓練，以避免未來資訊洩漏。'
    }
    print('\n測試結果：', json.dumps(report, ensure_ascii=False, indent=2))
    print(classification_report(
        y_test, pred, labels=list(range(n_classes)),
        target_names=labels, zero_division=0
    ))
    print('混淆矩陣（列為真實類別、欄為預測類別；順序：出局、一安、長打、保送）：')
    print(confusion_matrix(y_test, pred, labels=list(range(n_classes))))

    model.save(ROOT / 'cpbl_mlp_h2h_4class_class_weight.keras')
    joblib.dump({
        'preprocessor': prep,
        'categorical_columns': cats,
        'numeric_columns': nums,
        'labels': labels
    }, ROOT / 'cpbl_mlp_h2h_4class_class_weight_preprocessor.joblib')
    out = test[[
        '比賽日期', '打者', '投手', '打擊結果', '標籤',
        'h2h_prior_pa_count', 'h2h_out_rate', 'h2h_single_rate',
        'h2h_xbh_rate', 'h2h_walk_rate'
    ]].copy()
    out['預測結果'] = [labels[i] for i in pred]
    for i, name in enumerate(labels):
        out['機率_' + name] = probs[:, i]
    out.to_csv(ROOT / 'cpbl_mlp_h2h_4class_class_weight_predictions.csv', index=False, encoding='utf-8-sig')
    (ROOT / 'cpbl_mlp_h2h_4class_class_weight_metrics.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    print("Macro ROC-AUC:", auc)

    print("\n========== 全部 Epoch 訓練紀錄 ==========")
    for i in range(len(history.history['loss'])):
        print(
        f"Epoch {i+1} | "
        f"Accuracy: {history.history['accuracy'][i]:.4f} | "
        f"Loss: {history.history['loss'][i]:.4f} | "
        f"Val Accuracy: {history.history['val_accuracy'][i]:.4f} | "
        f"Val Loss: {history.history['val_loss'][i]:.4f}"
    )
   




if __name__ == '__main__':
    main()
