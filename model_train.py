#import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
np.random.seed(2)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
# from keras.optimizers import adam
from keras.callbacks import EarlyStopping

from PIL import Image, ImageChops, ImageEnhance
import os
import itertools

def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'

    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image

real_image_path = r"C:\Users\alurk\OneDrive\Desktop\CASIA2\Au\Au_ani_00001.jpg"
fake_image_path = r"C:\Users\alurk\OneDrive\Desktop\CASIA2\Tp\Tp_D_CND_M_N_ani00018_sec00096_00138.tif"

Image.open(real_image_path)
convert_to_ela_image(real_image_path, 90)

Image.open(fake_image_path)
convert_to_ela_image(fake_image_path, 90)

image_size = (128, 128)
def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0

X = [] # ELA converted images
Y = [] # 0 for fake, 1 for real

import random
path = r'C:\Users\alurk\OneDrive\Desktop\CASIA2'
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('jpg') or filename.endswith('png'):
            full_path = os.path.join(dirname, filename)
            X.append(prepare_image(full_path))
            Y.append(1)
            if len(Y) % 500 == 0:
                print(f'Processing {len(Y)} images')

random.shuffle(X)
X = X[:2100]
Y = Y[:2100]
print(len(X), len(Y))

path = r'C:\Users\alurk\OneDrive\Desktop\CASIA2\Tp'
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('jpg') or filename.endswith('png'):
            full_path = os.path.join(dirname, filename)
            X.append(prepare_image(full_path))
            Y.append(0)
            if len(Y) % 500 == 0:
                print(f'Processing {len(Y)} images')

print(len(X), len(Y))

X = np.array(X)
Y = to_categorical(Y, 2)
X = X.reshape(-1, 128, 128, 3)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=5)
X = X.reshape(-1, 1, 1, 3)
print(len(X_train), len(Y_train))
print(len(X_val), len(Y_val))

from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D

def build_model():
    # Load the pre-trained VGG16 model without the top layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    
    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add new top layers for fine-tuning
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation='softmax')(x)
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

model = build_model()
model.summary()

epochs = 5
batch_size = 32

init_lr = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr, decay=init_lr/epochs)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_accuracy',
                               min_delta=0,
                               patience=2,
                               verbose=0,
                               mode='auto')

hist = model.fit(X_train,
                 Y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 validation_data=(X_val, Y_val),
                 callbacks=[early_stopping])

model.save('my__model.keras')

def pred(image_path):
    image = prepare_image(image_path)
    image = image.reshape(-1, 128, 128, 3)
    y_pred = model.predict(image)
    y_pred_class = np.argmax(y_pred, axis=1)[0]
    print(f'Class: {class_names[y_pred_class]} Confidence: {np.amax(y_pred) * 100:0.2f}')
    
    # Plot the loss and accuracy curves for training and validation
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(hist.history['loss'], color='b', label="Training loss")
    ax[0].plot(hist.history['val_loss'], color='r', label="Validation loss", axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(hist.history['accuracy'], color='b', label="Training accuracy")
    ax[1].plot(hist.history['val_accuracy'], color='r', label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(Y_pred,axis = 1)
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1)
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(2))

class_names = ['fake', 'real']
fake_image = os.listdir(r'C:\Users\alurk\OneDrive\Desktop\CASIA2\Tp')
correct = 0
total = 0
for file_name in fake_image:
    if file_name.endswith('jpg') or filename.endswith('png'):
        fake_image_path = os.path.join(r'C:\Users\alurk\OneDrive\Desktop\CASIA2\Tp', file_name)
        image = prepare_image(fake_image_path)
        image = image.reshape(-1, 128, 128, 3)
        y_pred = model.predict(image)
        y_pred_class = np.argmax(y_pred, axis = 1)[0]
        total += 1
        if y_pred_class == 0:
            correct += 1
            print(f'Class: {class_names[y_pred_class]} Confidence: {np.amax(y_pred) * 100:0.2f}')
            print(f'Total: {total}, Correct: {correct}, Acc: {correct / total * 100.0}')

real_image = os.listdir(r'C:\Users\alurk\OneDrive\Desktop\CASIA2\Au')
correct_r = 0
total_r = 0
print(len(real_image))
i = 1
for file_name in real_image:
    if file_name.endswith('jpg') or filename.endswith('png'):
        real_image_path = os.path.join(r'C:\Users\alurk\OneDrive\Desktop\CASIA2\Au', file_name)
        image = prepare_image(real_image_path)
        image = image.reshape(-1, 128, 128, 3)
        y_pred = model.predict(image)
        y_pred_class = np.argmax(y_pred, axis = 1)[0]
        total_r += 1
        if y_pred_class == 1:
            correct_r += 1
            print (i)
            print(f'Class: {class_names[y_pred_class]} Confidence: {np.amax(y_pred) * 100:0.2f}')
            i = i + 1

correct += correct_r
total += total_r
print(f'Total: {total_r}, Correct: {correct_r}, Acc: {correct_r / total_r * 100.0}')
print(f'Total: {total}, Correct: {correct}, Acc: {correct / total * 100.0}')


