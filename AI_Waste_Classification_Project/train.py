from pathlib import Path
import argparse
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "split"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
IMG_SIZE = (224, 224)
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

def make_dataset(folder, image_size, batch_size, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        folder, image_size=image_size, batch_size=batch_size,
        label_mode="int", shuffle=shuffle, seed=SEED
    )

def build_model(num_classes):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False, weights="imagenet"
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.08)(x)
    x = tf.keras.layers.RandomZoom(0.10)(x)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, base

def plot_history(history, out_path):
    hist = history.history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(hist.get("accuracy", []), label="Train")
    axes[0].plot(hist.get("val_accuracy", []), label="Validation")
    axes[0].set_title("Model accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(hist.get("loss", []), label="Train")
    axes[1].plot(hist.get("val_loss", []), label="Validation")
    axes[1].set_title("Model loss"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not (DATA / "train").exists():
        raise SystemExit("Prepared dataset not found. Run python prepare_data.py first.")
    MODELS.mkdir(exist_ok=True); OUTPUTS.mkdir(exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)

    train_ds = make_dataset(DATA / "train", IMG_SIZE, args.batch_size, True)
    val_ds = make_dataset(DATA / "val", IMG_SIZE, args.batch_size, False)
    test_ds = make_dataset(DATA / "test", IMG_SIZE, args.batch_size, False)
    class_names = train_ds.class_names
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    model, base = build_model(len(class_names))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            MODELS / "waste_classifier.keras", monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6)
    ]
    print("\nPhase 1: training classification head")
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # Fine-tune only the final layers to adapt the pretrained features to waste images.
    print("\nPhase 2: fine-tuning final MobileNetV2 layers")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    history2 = model.fit(train_ds, validation_data=val_ds, epochs=max(3, args.epochs // 2),
                         callbacks=callbacks)

    best_model = tf.keras.models.load_model(MODELS / "waste_classifier.keras")
    loss, acc = best_model.evaluate(test_ds, verbose=0)
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    probs = best_model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(OUTPUTS / "classification_report.csv")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(OUTPUTS / "confusion_matrix.png", dpi=160); plt.close(fig)

    merged = {}
    for h in (history1.history, history2.history):
        for key, values in h.items():
            merged.setdefault(key, []).extend(values)
    class History: pass
    combined = History(); combined.history = merged
    plot_history(combined, OUTPUTS / "training_history.png")

    metrics = {
        "test_loss": float(loss), "test_accuracy": float(acc),
        "test_accuracy_percent": round(float(acc) * 100, 2),
        "test_images": int(len(y_true)), "class_names": class_names
    }
    (OUTPUTS / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    (MODELS / "class_names.json").write_text(json.dumps(class_names, indent=2))
    print("\nTEST RESULTS")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {MODELS / 'waste_classifier.keras'}")
    print(f"Saved evaluation outputs in: {OUTPUTS}")

if __name__ == "__main__":
    main()
