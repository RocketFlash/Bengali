import pandas as pd
import matplotlib.pyplot as plt
from classification_models.keras import Classifiers
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras_radam import RAdam
from PIL import Image
import keras
import numpy as np
import cv2
import os

def generator_wrapper(generator):
    n_classes = [168, 11, 7]
    for batch_x, batch_y in generator:
        yield (batch_x,[to_categorical(batch_y[:,i], num_classes=n_classes[i]) for i in range(3)])

DATASET_PATH = '/home/rauf/datasets/bengali/'
HEIGHT = 137
WIDTH = 236
SIZE = 128

train_csv_file = os.path.join(DATASET_PATH, 'train.csv')
full_df = pd.read_csv(train_csv_file)
full_df['image_id'] = full_df['image_id'].apply(lambda x: x + '.png') 
full_df.head()

train_df, val_df = train_test_split(full_df, test_size=0.2)

columns = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
images_path_train = os.path.join(DATASET_PATH, 'pngs/','train')
images_path_test = os.path.join(DATASET_PATH, 'pngs/','test')

datagen = ImageDataGenerator(rescale=1./255.)
test_datagen = ImageDataGenerator(rescale=1./255.)

batch_size = 64

train_generator = datagen.flow_from_dataframe(dataframe=train_df,
                                            directory=images_path_train,
                                            x_col="image_id",
                                            y_col=columns,
                                            batch_size=batch_size,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="other",
                                            target_size=(SIZE,SIZE))

val_generator = datagen.flow_from_dataframe(dataframe=val_df,
                                            directory=images_path_train,
                                            x_col="image_id",
                                            y_col=columns,
                                            batch_size=batch_size,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="other",
                                            target_size=(SIZE,SIZE))

# BaseModel, preprocess_input = Classifiers.get('resnext101')
BaseModel, preprocess_input = Classifiers.get('resnet18')
base_model = BaseModel((128, 128, 3), weights='imagenet', include_top=False)

n_classes_grapheme = 168
n_classes_vowel = 11
n_classes_consonant = 7

x = keras.layers.GlobalAveragePooling2D()(base_model.output)
output_grapheme = keras.layers.Dense(n_classes_grapheme, activation='softmax')(x)
output_vowel = keras.layers.Dense(n_classes_vowel, activation='softmax')(x)
output_consonant = keras.layers.Dense(n_classes_consonant, activation='softmax')(x)
model = keras.models.Model(inputs=[base_model.input], 
                           outputs=[output_grapheme, output_vowel, output_consonant])

model.compile(optimizer=RAdam(), loss=['categorical_crossentropy', 
                                     'categorical_crossentropy',
                                     'categorical_crossentropy'], metrics=['accuracy'])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=val_generator.n//val_generator.batch_size

model.fit_generator(generator=generator_wrapper(train_generator),
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=generator_wrapper(val_generator),
                    validation_steps=STEP_SIZE_VALID,
                    epochs=1,verbose=1)