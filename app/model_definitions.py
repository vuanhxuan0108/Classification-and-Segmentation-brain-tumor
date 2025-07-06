from tensorflow.keras.models import Model
from keras.applications import ResNet50
from tensorflow.keras import layers
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

def build_classifier_model(input_shape=(224, 224, 3), num_classes=4):
    base_model = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')

    for layer in base_model.layers[-25:]:
        layer.trainable = True

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model

def build_segmentation_model(input_shape=(256, 256, 3)):
    model = sm.Unet(
        backbone_name='efficientnetb0',
        encoder_weights='imagenet',
        input_shape=input_shape,
        classes=1,
        activation='sigmoid'
    )

    return model
