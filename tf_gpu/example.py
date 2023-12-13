import argparse
import json
import os
import shutil
import tensorflow as tf


parser = argparse.ArgumentParser(prog='example')
parser.add_argument('-i', '--input', required=True, help='Input file')
parser.add_argument('-m', '--model', required=True, help='Output file')


def make_archive(source, destination):
    base = os.path.basename(destination)
    name = base.split(".")[0]
    format = base.split(".")[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, format, archive_from, archive_to)
    shutil.move(f"{name}.{format}", destination)


def main(epoch_count, model_file):
    print("TensorFlow version: ", tf.__version__)
    print("")
    print(os.system("nvidia-smi"))
    print("")

    print("Load MNIST dataset...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print("Build Sequential model...")
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation="relu"),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10)
    ])

    print("Compile model...")
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    print("Fit...")
    model.fit(x_train, y_train, epochs=epoch_count)

    print("Evaluate...")
    model.evaluate(x_test,  y_test, verbose=2)

    print(f"Save model to '{model_file}'")
    tf.keras.models.save_model(model, "model", save_format="tf")
    make_archive("model", model_file)

    print("Done")


if __name__ == "__main__":
    args = parser.parse_args()

    epoch_count = 5

    with open(args.input) as f:
        data = json.load(f)
        epoch_count = int(data["epoch_count"])

    main(epoch_count, args.model)

