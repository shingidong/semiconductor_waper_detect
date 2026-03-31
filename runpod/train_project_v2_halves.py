from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model(num_classes: int) -> Sequential:
    model = Sequential()
    model.add(layers.Input(shape=(64, 64, 3)))
    model.add(layers.Conv2D(filters=32, kernel_size=(4, 4), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(rate=0.3))

    model.add(layers.Conv2D(filters=32, kernel_size=(4, 4), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(rate=0.3))

    model.add(layers.Conv2D(filters=32, kernel_size=(4, 4), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(rate=0.3))

    model.add(layers.Conv2D(filters=32, kernel_size=(4, 4), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(rate=0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_generators(data_root: Path, image_size: tuple[int, int], batch_size: int):
    train_dir = data_root / "train"
    valid_dir = data_root / "valid"
    test_dir = data_root / "test"

    for path in (train_dir, valid_dir, test_dir):
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset path: {path}")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=image_size,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True,
    )
    valid_gen = eval_datagen.flow_from_directory(
        str(valid_dir),
        target_size=image_size,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
    )
    test_gen = eval_datagen.flow_from_directory(
        str(test_dir),
        target_size=image_size,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False,
    )
    return train_gen, valid_gen, test_gen


def save_training_curve(history, output_dir: Path) -> None:
    history_dict = history.history
    epochs = range(1, len(history_dict["loss"]) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_dict["loss"], label="train_loss")
    plt.plot(epochs, history_dict["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_dict["accuracy"], label="train_acc")
    plt.plot(epochs, history_dict["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy")

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close()


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], output_dir: Path) -> None:
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()


def train_and_evaluate(
    data_root: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    image_size: tuple[int, int],
    initial_weights: Path | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_gen, valid_gen, test_gen = build_generators(data_root, image_size, batch_size)
    class_names = [name for name, _idx in sorted(train_gen.class_indices.items(), key=lambda item: item[1])]

    model = build_model(num_classes=train_gen.num_classes)
    if initial_weights is not None:
        if not initial_weights.exists():
            raise FileNotFoundError(f"Initial weights not found: {initial_weights}")
        model.load_weights(str(initial_weights))
        print(f"Loaded initial weights from: {initial_weights}")

    checkpoint_path = output_dir / "best_model.keras"
    callbacks = [
        ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        CSVLogger(str(output_dir / "training_log.csv")),
    ]

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=valid_gen,
        callbacks=callbacks,
        verbose=1,
    )

    model = load_model(str(checkpoint_path))
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    probs = model.predict(test_gen, verbose=1)
    y_true = test_gen.classes
    y_pred = np.argmax(probs, axis=1)

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred)

    save_training_curve(history, output_dir)
    save_confusion_matrix(cm, class_names, output_dir)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_root": str(data_root),
                "class_indices": train_gen.class_indices,
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
                "classification_report": report_dict,
                "confusion_matrix": cm.tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== Final Test Metrics ===")
    print("test_loss:", float(test_loss))
    print("test_accuracy:", float(test_accuracy))
    print(report_text)
    print(f"Saved outputs to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the project_v2-style CNN on one dataset half.")
    parser.add_argument("--data-root", required=True, help="Path to half_1 or half_2 dataset root")
    parser.add_argument("--output-dir", required=True, help="Directory to save checkpoints and reports")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--image-size", type=int, default=64, help="Square image size")
    parser.add_argument("--initial-weights", default="", help="Optional .h5/.keras weights file to warm start from")
    args = parser.parse_args()

    initial_weights = Path(args.initial_weights).expanduser().resolve() if args.initial_weights else None
    train_and_evaluate(
        data_root=Path(args.data_root).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        initial_weights=initial_weights,
    )


if __name__ == "__main__":
    main()
