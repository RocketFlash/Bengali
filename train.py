import pandas as pd
import matplotlib.pyplot as plt
from classification_models.keras import Classifiers
import efficientnet.keras as efn
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras_radam import RAdam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from ImageDataAugmentor.image_data_augmentor import *
from keras import regularizers, optimizers
from PIL import Image
import keras
import numpy as np
import cv2
import os
from albumentations import (
    CLAHE, Rotate,
    Blur, GridDistortion, 
    GaussNoise, MotionBlur,  
    RandomBrightnessContrast, OneOf, Compose, Downscale,
    ElasticTransform
)

def generator_wrapper(generator):
    n_classes = [168, 11, 7]
    for batch_x, batch_y in generator:
        yield (batch_x,[to_categorical(batch_y[:,i], num_classes=n_classes[i]) for i in range(3)])

def plot_param(history, save_path, param='', metric='loss'):
    name = '{}_{}'.format(param, metric) if len(param)>0 else metric
    
    plt.plot(history.history['{}'.format(name)])
    plt.plot(history.history['val_{}'.format(name)])
    
    plt.title('Model {}'.format(name))
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig('{}{}.png'.format(save_path,name))

def plot_history(history, save_path):
    # plot total loss
    plot_param(history, save_path, param='', metric='loss')
    # plot losses
    plot_param(history, save_path, param='output_grapheme', metric='loss')
    plot_param(history, save_path, param='output_vowel', metric='loss')
    plot_param(history, save_path, param='output_consonant', metric='loss')
    # plot accuracies
    plot_param(history, save_path, param='output_grapheme', metric='acc')
    plot_param(history, save_path, param='output_vowel', metric='acc')
    plot_param(history, save_path, param='output_consonant', metric='acc')
    

# PARAMETERS
# ==================================================================
DATASET_PATH = '/home/rauf/datasets/bengali/'
HEIGHT = 137
WIDTH = 236
SIZE = 128

n_epochs = 1000
batch_size = 32

work_dirs_path = 'work_dirs/'

backbone_name = 'efficientnet-b5'
config_name = backbone_name + '_augmentations_albumentations'
weights_save_path = os.path.join(work_dirs_path, config_name, 'weights/')
logs_save_path = os.path.join(work_dirs_path, config_name, 'logs/')
plots_save_path = os.path.join(work_dirs_path, config_name, 'plots/')
os.makedirs(work_dirs_path, exist_ok=True)
os.makedirs(weights_save_path, exist_ok=True)
os.makedirs(logs_save_path, exist_ok=True)
os.makedirs(plots_save_path, exist_ok=True)

n_classes_grapheme = 168
n_classes_vowel = 11
n_classes_consonant = 7
# optimizer = optimizers.rmsprop(lr = 0.0001, decay = 1e-6)
optimizer = RAdam(learning_rate=0.001)

AUGMENTATIONS = Compose([OneOf([
    Blur(always_apply=False, p=1.0, blur_limit=(3, 7)),
    Downscale(always_apply=False, p=1.0, scale_min=0.3, scale_max=0.5, interpolation=0),
    ElasticTransform(always_apply=False, p=1.0, alpha=0.6, sigma=16, alpha_affine=45, interpolation=0, border_mode=1),
    GaussNoise(always_apply=False, p=1.0, var_limit=(10.0, 115.12999725341797)),
    GridDistortion(always_apply=False, p=1.0, num_steps=6, distort_limit=(-0.5, 0.5), interpolation=0, border_mode=1),
    MotionBlur(always_apply=False, p=1.0, blur_limit=(3, 20)),
    Rotate(always_apply=False, p=1.0, limit=(-28, 28), interpolation=0, border_mode=1)
    ], p=0.9)
])
# ==================================================================


train_csv_file = os.path.join(DATASET_PATH, 'train.csv')
full_df = pd.read_csv(train_csv_file)
full_df['image_id'] = full_df['image_id'].apply(lambda x: x + '.png') 
full_df.head()

train_df, val_df = train_test_split(full_df, test_size=0.1)

columns = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
images_path_train = os.path.join(DATASET_PATH, 'pngs/','train')
images_path_test = os.path.join(DATASET_PATH, 'pngs/','test')

# data_generator_dict = dict(rescale=1./255.,
#                             # featurewise_center=False,
#                             # samplewise_center=False,
#                             rotation_range=45,
#                             width_shift_range=0.1,
#                             height_shift_range=0.1,
#                             shear_range=0.2,
#                             zoom_range=[0.5, 1.5],
#                             fill_mode='reflect')

# datagen = ImageDataGenerator(**data_generator_dict)

datagen =  ImageDataAugmentor(rescale=1./255, augment = AUGMENTATIONS, preprocess_input=None)

train_generator = datagen.flow_from_dataframe(dataframe=train_df,
                                            directory=images_path_train,
                                            color_mode='gray',
                                            x_col="image_id",
                                            y_col=columns,
                                            batch_size=batch_size,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="other",
                                            target_size=(SIZE,SIZE))

val_generator = datagen.flow_from_dataframe(dataframe=val_df,
                                            directory=images_path_train,
                                            color_mode='gray',
                                            x_col="image_id",
                                            y_col=columns,
                                            batch_size=batch_size,
                                            seed=42,
                                            shuffle=True,
                                            class_mode="other",
                                            target_size=(SIZE,SIZE))

if backbone_name.startswith('efficientnet'):
    efficientnet_models = {
        'efficientnet-b0': efn.EfficientNetB0,
        'efficientnet-b1': efn.EfficientNetB1,
        'efficientnet-b2': efn.EfficientNetB2,
        'efficientnet-b3': efn.EfficientNetB3,
        'efficientnet-b4': efn.EfficientNetB4,
        'efficientnet-b5': efn.EfficientNetB5,
        'efficientnet-b6': efn.EfficientNetB6,
        'efficientnet-b7': efn.EfficientNetB7,
    }
    Efficientnet_model = efficientnet_models[backbone_name]
    base_model = Efficientnet_model(input_shape=(128, 128, 1), weights=None, include_top=False)
else:
    BaseModel, preprocess_input = Classifiers.get(backbone_name)
    base_model = BaseModel(input_shape=(128, 128, 1), weights=None, include_top=False)


# x0 = keras.layers.Flatten()(base_model.output)
x0 = keras.layers.GlobalAveragePooling2D()(base_model.output)

x = keras.layers.Dense(1024, activation='relu')(x0)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(rate=0.5)(x)
x = keras.layers.Dense(512, activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(rate=0.5)(x)
x = keras.layers.Dense(256, activation='relu')(x)
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

x = keras.layers.Dense(512, activation='relu')(x0)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(rate=0.5)(x)
x = keras.layers.Dense(256, activation='relu')(x)
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

x = keras.layers.Dense(512, activation='relu')(x0)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(rate=0.5)(x)
x = keras.layers.Dense(256, activation='relu')(x)
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
                                                 'output_vowel': 0.08,
                                                 'output_consonant': 0.08}, 
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
             TensorBoard(log_dir=tensorboard_save_name),
             EarlyStopping(monitor='val_loss',
                      patience=15, 
                      verbose=1),
             ReduceLROnPlateau(monitor='val_loss', 
                               factor=0.1,
                               patience=5, 
                               verbose=1)]

# model.load_weights(checkpoints_save_name)

history = model.fit_generator(generator=generator_wrapper(train_generator),
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=generator_wrapper(val_generator),
                    validation_steps=STEP_SIZE_VALID,
                    epochs=n_epochs,
                    verbose=1,
                    callbacks=callbacks)


plot_history(history, plots_save_path)