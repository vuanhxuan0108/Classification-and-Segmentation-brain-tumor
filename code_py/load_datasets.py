import tensorflow as tf
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import segmentation_models as sm

preprocess_input = sm.get_preprocessing('efficientnetb1')

def load_datasets_classification(train_path, test_path, batch_size=32, validation_split = 0.15, img_size=(224, 224), label_mode="categorical"):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        image_size=img_size,
        batch_size=batch_size,
        validation_split=validation_split, 
        subset="training",
        label_mode=label_mode,
        seed=42,
        shuffle=True 
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        image_size=img_size,
        batch_size=batch_size,
        validation_split=validation_split,
        subset="validation",
        label_mode=label_mode,
        seed=42,
        shuffle=True
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_path,
        image_size=img_size,
        batch_size=batch_size,
        label_mode=label_mode,
        shuffle=False
    )

    class_names = train_ds.class_names
    
    print(f"Class names: {class_names}")
    return train_ds, val_ds, test_ds, class_names

    
def create_df(base_dir):
    img_dir = os.path.join(base_dir, 'image')
    mask_dir = os.path.join(base_dir, 'mask')
    files = sorted(f for f in os.listdir(img_dir) if os.path.exists(os.path.join(mask_dir, f)))
    return pd.DataFrame({
        'image_paths': [os.path.join(img_dir, f) for f in files],
        'mask_paths': [os.path.join(mask_dir, f) for f in files]
    })


def split_df(df, train_size=0.7, valid_size=0.1, test_size=0.2):
    train_df, dummy_df = train_test_split(df, train_size= train_size, shuffle=True, random_state=28)

    valid_size_adjusted = valid_size / (valid_size + test_size)

    valid_df, test_df = train_test_split(dummy_df, train_size= valid_size_adjusted, shuffle=True, random_state=28)

    return train_df, valid_df, test_df

def preprocess_with_sm(image):
    image = image.numpy()
    image = preprocess_input(image)
    return image.astype(np.float32)

def tf_preprocess_with_sm(image):
    image = tf.py_function(preprocess_with_sm, [image], tf.float32)
    image.set_shape([256, 256, 3])
    return image

def parse_function(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (256, 256))
    img = tf_preprocess_with_sm(img)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (256, 256), method='nearest')
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.where(mask > 0.5, 1.0, 0.0)

    return img, mask

def create_dataset(df, batch_size=32, shuffle=True):
    img_paths = df['image_paths'].values
    mask_paths = df['mask_paths'].values

    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def load_datasets_segmentation(base_dir, batch_size=32, train_size=0.7, valid_size=0.1, test_size=0.2):
    df = create_df(base_dir)
    train_df, valid_df, test_df = split_df(df, train_size=train_size, valid_size=valid_size, test_size=test_size)

    train_ds = create_dataset(train_df, batch_size=batch_size, shuffle=True)
    valid_ds = create_dataset(valid_df, batch_size=batch_size, shuffle=False)
    test_ds = create_dataset(test_df, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_df)}, Validation samples: {len(valid_df)}, Test samples: {len(test_df)}")

    return train_df, test_df, valid_df, train_ds, valid_ds, test_ds