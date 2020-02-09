import pandas as pd
import matplotlib.pyplot as plt
from classification_models.keras import Classifiers
import efficientnet.keras as efn
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras_radam import RAdam
from keras.callbacks.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras import regularizers, optimizers
from PIL import Image
import keras
import numpy as np
import cv2
import os

def generator_wrapper(generator):
    n_classes = [168, 11, 7]
    for batch_x, batch_y in generator:
        # print([to_categorical(batch_y[:,i], num_classes=n_classes[i]) for i in range(3)])
        yield (batch_x,[to_categorical(batch_y[:,i], num_classes=n_classes[i]) for i in range(3)])

DATASET_PATH = '/home/rauf/datasets/bengali/'
HEIGHT = 137
WIDTH = 236
SIZE = 128

train_csv_file = os.path.join(DATASET_PATH, 'train.csv')
full_df = pd.read_csv(train_csv_file)
full_df['image_id'] = full_df['image_id'].apply(lambda x: x + '.png') 
full_df.head()

train_df, val_df = train_test_split(full_df, test_size=0.1)

columns = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
images_path_train = os.path.join(DATASET_PATH, 'pngs/','train')
images_path_test = os.path.join(DATASET_PATH, 'pngs/','test')

datagen = ImageDataGenerator(rescale=1./255.)
test_datagen = ImageDataGenerator(rescale=1./255.)

n_epochs = 1000
batch_size = 32

train_generator = datagen.flow_from_dataframe(dataframe=train_df,
                                            directory=images_path_train,
                                            color_mode='grayscale',
                                            x_col="image_id",
                                            y_col=columns,
                                            batch_size=batch_size,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="other",
                                            target_size=(SIZE,SIZE))

val_generator = datagen.flow_from_dataframe(dataframe=val_df,
                                            directory=images_path_train,
                                            color_mode='grayscale',
                                            x_col="image_id",
                                            y_col=columns,
                                            batch_size=batch_size,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="other",
                                            target_size=(SIZE,SIZE))

backbone_name = 'efficientnetB5'
# backbone_name = 'seresnet50'
weights_save_path = 'weights/'
logs_save_path = 'logs/'
plots_save_path = 'plots/'
# BaseModel, preprocess_input = Classifiers.get(backbone_name)
# base_model = BaseModel(input_shape=(128, 128, 1), weights=None, include_top=False)
base_model = efn.EfficientNetB5(input_shape=(128, 128, 1), weights=None, include_top=False)

n_classes_grapheme = 168
n_classes_vowel = 11
n_classes_consonant = 7
# optimizer = optimizers.rmsprop(lr = 0.0001, decay = 1e-6)
optimizer = RAdam(learning_rate=0.001)

# x0 = keras.layers.Flatten()(base_model.output)
x0 = keras.layers.GlobalAveragePooling2D()(base_model.output)
print(x0)
x = keras.layers.Dense(256, activation='relu')(x0)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(rate=0.5)(x)
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(rate=0.5)(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(rate=0.5)(x)
output_grapheme = keras.layers.Dense(n_classes_grapheme, 
                                     activation='softmax', 
                                     name = 'output_grapheme')(x)
x = keras.layers.Dense(256, activation='relu')(x0)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(rate=0.5)(x)
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(rate=0.5)(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(rate=0.5)(x)
output_vowel = keras.layers.Dense(n_classes_vowel, 
                                  activation='softmax', 
                                  name = 'output_vowel')(x)
x = keras.layers.Dense(256, activation='relu')(x0)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(rate=0.5)(x)
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(rate=0.5)(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(rate=0.5)(x)
output_consonant = keras.layers.Dense(n_classes_consonant, 
                                      activation='softmax', 
                                      name = 'output_consonant')(x)

model = keras.models.Model(inputs=[base_model.input], 
                           outputs=[output_grapheme, output_vowel, output_consonant])
model.summary()

model.compile(optimizer=optimizer, loss = {'output_grapheme':'categorical_crossentropy', 
                                         'output_vowel':'categorical_crossentropy',
                                         'output_consonant':'categorical_crossentropy'},
                                 loss_weights = {'output_grapheme': 1,
                                                 'output_vowel': 0.2,
                                                 'output_consonant': 0.2}, 
                                 metrics=['accuracy'])

# keras.utils.plot_model(model, '{}{}_model.png'.format(plots_save_path, backbone_name))

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=val_generator.n//val_generator.batch_size

checkpoints_save_name = '{}best_{}.hdf5'.format(weights_save_path,backbone_name)
tensorboard_save_name = '{}logs_{}/'.format(logs_save_path,backbone_name)

callbacks = [ModelCheckpoint(checkpoints_save_name , 
                monitor='val_loss',
                save_best_only=True, 
                verbose=1),
            #  TensorBoard(log_dir=tensorboard_save_name, 
            #              histogram_freq=0),
             EarlyStopping(monitor='val_loss',
                      patience=15, 
                      verbose=1),
             ReduceLROnPlateau(monitor='val_loss', 
                               factor=0.1,
                               patience=5, 
                               verbose=1)]

# model.load_weights(checkpoints_save_name)

model.fit_generator(generator=generator_wrapper(train_generator),
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=generator_wrapper(val_generator),
                    validation_steps=STEP_SIZE_VALID,
                    epochs=n_epochs,
                    use_multiprocessing=True,
                    verbose=1,
                    callbacks=callbacks)
