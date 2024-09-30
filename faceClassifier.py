import shutil
import random
import os

dataset_path = '/kaggle/input/human-faces-dataset/Human Faces Dataset'
real_images_path = os.path.join(dataset_path, 'Real Images')
ai_generated_images_path = os.path.join(dataset_path, 'AI-Generated Images')

output_path = '/kaggle/working/'
train_path = os.path.join(output_path, 'train')
val_path = os.path.join(output_path, 'val')
test_path = os.path.join(output_path, 'test')

for path in [train_path, val_path, test_path]:
    os.makedirs(os.path.join(path, 'Real Images'), exist_ok=True)
    os.makedirs(os.path.join(path, 'AI-Generated Images'), exist_ok=True)

train_split = 0.7
val_split = 0.15
test_split = 0.15

def split_and_copy_images(source_dir, dest_dirs, split_ratios):
    images = os.listdir(source_dir)
    random.shuffle(images)
    
    train_size = int(len(images) * split_ratios[0])
    val_size = int(len(images) * split_ratios[1])
    
    for i, img in enumerate(images):
        if i < train_size:
            dest_dir = dest_dirs[0]
        elif i < train_size + val_size:
            dest_dir = dest_dirs[1]
        else:
            dest_dir = dest_dirs[2]
            
        shutil.copy(os.path.join(source_dir, img), os.path.join(dest_dir, img))

split_and_copy_images(real_images_path, [os.path.join(train_path, 'Real Images'), os.path.join(val_path, 'Real Images'), os.path.join(test_path, 'Real Images')], [train_split, val_split, test_split])
split_and_copy_images(ai_generated_images_path, [os.path.join(train_path, 'AI-Generated Images'), os.path.join(val_path, 'AI-Generated Images'), os.path.join(test_path, 'AI-Generated Images')], [train_split, val_split, test_split])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_height, img_width = 224, 224
batch_size = 32

train_datagen = ImageDataGenerator(rescale = 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = 'binary'
)

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = 'binary'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = 'binary'
)

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation ='relu')(x)
predictions = Dense(1, activation ='sigmoid')(x)

model = Model(inputs = base_model.input, outputs = predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit(
    train_generator,
    validation_data = val_generator,
    epochs = 10
)

test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def show_predictions_table_with_correct_labels(model, generator, num_images = 20, threshold = 0.5):
    images, true_labels = next(generator)
    images, true_labels = images[:num_images], true_labels[:num_images]
    
    predictions = model.predict(images)
    predicted_labels = (predictions >= threshold).astype(int)
  
    if true_labels.ndim > 1:
        true_labels = np.argmax(true_labels, axis=1)
    
    true_label_names = ["Real" if label == 1 else "AI Generated" for label in true_labels]
    predicted_label_names = ["Real" if label == 1 else "AI Generated" for label in predicted_labels]
    correct_predictions = predicted_labels.flatten() == true_labels.flatten()

    df = pd.DataFrame({
        "Image Index": list(range(num_images)),
        "True Label": true_label_names,
        "Predicted Label": predicted_label_names,
        "Correct": correct_predictions
    })

    fig, axes = plt.subplots(5, 4, figsize = (15, 12))

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_title(f"True: {df['True Label'][i]}\nPred: {df['Predicted Label'][i]}\nCorrect: {df['Correct'][i]}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    return df

show_predictions_table_with_correct_labels(model, val_generator, num_images = 20, threshold = 0.5)
