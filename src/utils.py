import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from collections import Counter
import contextlib
import os
from tensorflow import keras
from IPython.display import Image, display
import matplotlib as mpl


def plot_sample_images(train_ds, class_names):
    class_samples = {class_name: [] for class_name in class_names}

    for images, labels in train_ds.unbatch():
        label = labels.numpy()
        label = np.argmax(label)
        class_name = class_names[label]

        image = images.numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype("uint8")
        else:
            image = image.astype("uint8")

        if len(class_samples[class_name]) < 3:
            class_samples[class_name].append(image)

        if all(len(imgs) == 3 for imgs in class_samples.values()):
            break

    plt.figure(figsize=(12, 9))
    for col_idx, (class_name, images) in enumerate(class_samples.items()):
        for row_idx in range(3):
            ax = plt.subplot(3, 4, row_idx * 4 + col_idx + 1)
            plt.imshow(images[row_idx])
            if row_idx == 0:
                plt.title(class_name, fontsize=12)
            plt.axis("off")

    plt.tight_layout()
    plt.show()
    
@tf.autograph.experimental.do_not_convert
def count_class_distribution(dataset, class_names):
    all_labels = dataset.map(lambda x, y: y)
    all_labels = tf.concat(list(all_labels), axis=0)
    class_counts = Counter(np.argmax(all_labels.numpy(), axis=1))
    return {class_names[i]: count for i, count in class_counts.items()}

def plot_distribution(df, title, palette_name="Blues", x_col="Class", hue_col="Class"):
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    sorted_df = df.sort_values("Count")
    labels = sorted_df[x_col].tolist()
    sizes = sorted_df["Count"].tolist()

    colors = sns.color_palette(palette_name, len(sizes))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90, wedgeprops={'edgecolor': 'black'})
    plt.title(title)

    plt.subplot(2, 2, 2)
    sns.barplot(x=x_col, y="Count", data=sorted_df, palette=colors, hue=hue_col, dodge=False, order=labels)

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel("Number of Images")
    plt.xticks(rotation=20)

    plt.tight_layout()
    plt.show()

def plot_predicted_images(model, test_ds, class_names):

    class_samples = {class_name: [] for class_name in class_names}

    for images, labels in test_ds.unbatch():
        image = images.numpy()
        label = labels.numpy()
        label = np.argmax(label)
        class_name = class_names[label]

        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)
        predicted_label = np.argmax(prediction, axis=1)[0]
        predicted_class_name = class_names[predicted_label]

        if len(class_samples[class_name]) < 3:
            class_samples[class_name].append((image, predicted_class_name))

        if all(len(imgs) == 3 for imgs in class_samples.values()):
            break

    plt.figure(figsize=(12, 9))
    for col_idx, (class_name, samples) in enumerate(class_samples.items()):
        for row_idx in range(3):
            image, predicted_class_name = samples[row_idx]
            if image.max() <= 1.0:
                image_display = (image * 255).astype("uint8")
            else:
                image_display = image.astype("uint8")

            ax = plt.subplot(3, 4, row_idx * 4 + col_idx + 1)
            plt.imshow(image_display)
            if row_idx == 0:
                plt.title(f"True: {class_name}\nPred: {predicted_class_name}", fontsize=12)
            else:
                plt.title(f"Pred: {predicted_class_name}", fontsize=12)
            plt.axis("off")

    plt.tight_layout()
    plt.show()