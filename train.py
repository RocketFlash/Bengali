import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from classification_models.tfkeras import Classifiers
import efficientnet.tfkeras as efn
from sklearn.model_selection import train_test_split
from ImageDataAugmentor.image_data_augmentor import *
from PIL import Image
import numpy as np
import cv2
import os
import gc
os.environ["TF_KERAS"] = "1"
from albumentations import (
    CLAHE, Rotate,
    Blur, GridDistortion, 
    GaussNoise, MotionBlur,  
    RandomBrightnessContrast, OneOf, Compose, Downscale,
    ElasticTransform
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, GlobalAveragePooling2D, Conv2D, Input
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.layers import Activation, Lambda, Layer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
# from tensorflow.keras.utils.generic_utils import get_custom_objects
# from tensorflow.keras_preprocessing.image import ImageDataGenerator
from keras_radam import RAdam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers, optimizers
# from tensorflow.keras.engine import Layer
from tensorflow.keras import backend as K
import albumentations

from keras_adabound import AdaBound
from utils import plot_history, CustomReduceLRonPlateau, GridMask, GroupNormalization, get_class_weights_dict
from utils import generator_wrapper, RandomAugMix, focal_loss

# CustomReduceLRonPlateau function
best_val_loss = np.Inf

# PARAMETERS
# ==================================================================
DATASET_PATH = '/home/rauf/datasets/bengali/'
HEIGHT = 137
WIDTH = 236
SIZE = 128
BATCH_SIZE = 32

columns = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
output_names = ['output_grapheme', 'output_vowel', 'output_consonant']
images_path_train = os.path.join(DATASET_PATH, 'pngs/','train')
images_path_test = os.path.join(DATASET_PATH, 'pngs/','test')

freeze_backbone = False
group_norm = True


work_dirs_path = 'work_dirs/'

backbone_name = 'efficientnet-b3'
config_name = backbone_name + '_k_fold'
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
# optimizer = optimizers.Adam(lr = 0.00016)
# optimizer = RAdam(learning_rate=0.00016)
optimizer = AdaBound(lr=0.00001,
                     final_lr=.1)

AUGMENTATIONS = Compose([GridMask(num_grid=(7,9), rotate=90,mode=0,p=1),
    OneOf([ RandomAugMix(severity=1, width=1, p=1.),
    # Downscale(always_apply=False, p=0.5, scale_min=0.3, scale_max=0.5, interpolation=0),
    # ElasticTransform(always_apply=False, p=0.2, alpha=0.3, sigma=10, alpha_affine=25, interpolation=0, border_mode=1),
    # GridDistortion(always_apply=False, p=0.2, num_steps=6, distort_limit=(-0.15, 0.15)),
    # MotionBlur(always_apply=False, p=0.5, blur_limit=(3, 20)),
    ], p=0.95)
])
# ==================================================================


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
    base_model = Efficientnet_model(input_shape=(128, 128, 3), 
                                    weights='noisy-student', 
                                    include_top=False)
    # if freeze_backbone:
    #     for layer in base_model.layers[:-2]:
    #         layer.trainable = False

    #checkpoints_load_name = 'work_dirs/efficientnet-b5_aug_alb_balance_swish_imagenet_small/weights/best_efficientnet-b5.hdf5'
    #base_model.load_weights(checkpoints_load_name, by_name=True)
else:
    BaseModel, preprocess_input = Classifiers.get(backbone_name)
    base_model = BaseModel(input_shape=(128, 128, 3), weights='imagenet', include_top=False)

if freeze_backbone:
    for layer in base_model.layers[:-2]:
        layer.trainable = False
else:
    for layer in base_model.layers:
            layer.trainable = True

if group_norm:
    for i, layer in enumerate(base_model.layers):
        if "batch_normalization" in layer.name:
            base_model.layers[i] = GroupNormalization(groups=32, 
                                                axis=-1, 
                                                epsilon=0.00001)
activation_function_grapheme = 'swish'
activation_function_vowel = 'swish'
activation_function_consonant = 'swish'

# input_layer = Input(shape=(128, 128, 1))
# conv_x = Conv2D(3, (3,3),padding='same')(input_layer)
# x0 = base_model(conv_x)
# x0 = keras.layers.Flatten()(base_model.output)
x0 = GlobalAveragePooling2D()(base_model.output)


# x = Dense(1024, activation=activation_function_grapheme, name = 'dense_grapheme_1')(x0)
# x = BatchNormalization()(x)
# x = Dropout(rate=0.5)(x)
# x = Dense(256, activation=activation_function_grapheme, name = 'dense_grapheme_2')(x0)
# x = BatchNormalization()(x)
# x = Dropout(rate=0.5)(x)
# x = Dense(256, activation=activation_function_grapheme, name = 'dense_grapheme_3')(x)
# x = BatchNormalization()(x)
# x = Dropout(rate=0.5)(x)
output_grapheme = Dense(n_classes_grapheme, 
                                     activation='softmax', 
                                     name = 'output_grapheme')(x0)

# x = Dense(512, activation=activation_function_vowel, name = 'dense_vowel_1')(x0)
# x = BatchNormalization()(x)
# x = Dropout(rate=0.5)(x)
# x = Dense(128, activation=activation_function_vowel, name = 'dense_vowel_2')(x0)
# x = BatchNormalization()(x)
# x = Dropout(rate=0.5)(x)
# x = Dense(128, activation=activation_function_vowel, name = 'dense_vowel_3')(x)
# x = BatchNormalization()(x)
# x = Dropout(rate=0.5)(x)
# x = Dense(64, activation=activation_function_vowel, name = 'dense_vowel_4')(x)
# x = BatchNormalization()(x)
# x = Dropout(rate=0.5)(x)
output_vowel = Dense(n_classes_vowel, 
                                  activation='softmax', 
                                  name = 'output_vowel')(x0)

# x = Dense(512, activation=activation_function_consonant, name = 'dense_consonant_1')(x0)
# x = BatchNormalization()(x)
# x = Dropout(rate=0.5)(x)
# x = Dense(128, activation=activation_function_consonant, name = 'dense_consonant_2')(x0)
# x = BatchNormalization()(x)
# x = Dropout(rate=0.5)(x)
# x = Dense(128, activation=activation_function_consonant, name = 'dense_consonant_3')(x)
# x = BatchNormalization()(x)
# x = Dropout(rate=0.5)(x)
# x = Dense(64, activation=activation_function_consonant, name = 'dense_consonant_4')(x)
# x = BatchNormalization()(x)
# x = Dropout(rate=0.5)(x)
output_consonant = Dense(n_classes_consonant, 
                                      activation='softmax', 
                                      name = 'output_consonant')(x0)


model = Model(inputs=[base_model.input], 
                           outputs=[output_grapheme, output_vowel, output_consonant])
model.summary()

model.compile(optimizer=optimizer, loss = {'output_grapheme':focal_loss(alpha=.25, gamma=2), 
                                         'output_vowel':focal_loss(alpha=.25, gamma=2),
                                         'output_consonant':focal_loss(alpha=.25, gamma=2)},
                                 loss_weights = {'output_grapheme': 1,
                                                 'output_vowel': 0.3,
                                                 'output_consonant': 0.3}, 
                                 metrics={'output_grapheme': ['accuracy', tf.keras.metrics.Recall()],
                                          'output_vowel': ['accuracy', tf.keras.metrics.Recall()],
                                          'output_consonant': ['accuracy', tf.keras.metrics.Recall()] })

model.save_weights("model_initial.h5")



train_csv_file = os.path.join(DATASET_PATH, 'train.csv')
full_df = pd.read_csv(train_csv_file)
full_df['image_id'] = full_df['image_id'].apply(lambda x: x + '.png') 
full_df.head()

# grapheme2idx = {grapheme: idx for idx, grapheme in enumerate(full_df.grapheme.unique())}
# full_df['grapheme_id'] = full_df['grapheme'].map(grapheme2idx)

# n_fold = 5
# skf = StratifiedKFold(n_fold, random_state=42)
# for i_fold, (train_idx, val_idx) in enumerate(skf.split(full_df, full_df.grapheme)):
#     full_df.loc[val_idx, 'fold'] = i_fold
# full_df['fold'] = full_df['fold'].astype(int)

# full_df['unseen'] = 0
# full_df.loc[full_df.grapheme_id >= 1245, 'unseen'] = 1

# full_df.loc[full_df['unseen'] == 1, 'fold'] = -1

X_train = full_df['image_id'].values
f_df = full_df[columns].astype('uint8')
for col in columns:
    f_df[col] = f_df[col].map('{:03}'.format)
Y_train = pd.get_dummies(f_df)
train_df, val_df = train_test_split(full_df, test_size=0.05)

EPOCHS = 7
TEST_SIZE = 1./6
msss = MultilabelStratifiedShuffleSplit(n_splits = EPOCHS, test_size = TEST_SIZE, random_state = 42)
datagen =  ImageDataAugmentor(rescale=1./255, augment = AUGMENTATIONS, preprocess_input=None)
# msss = MultilabelStratifiedKFold
n_epochs = 20
checkpoints_load_name = 'work_dirs/efficientnet-b3_k_fold/weights/_efficientnet-b3__k_fold_shuffle_ep_0_019_0.9138.hdf5'

for epoch_n, msss_splits in zip(range(0, EPOCHS), msss.split(X_train, Y_train)):
    # model.load_weights("model_initial.h5")
    model.load_weights(checkpoints_load_name, by_name=True)
    train_idx = msss_splits[0]
    valid_idx = msss_splits[1]
    np.random.shuffle(train_idx)
    print('Train Length: {0}   First 10 indices: {1}'.format(len(train_idx), train_idx[:10]))    
    print('Valid Length: {0}    First 10 indices: {1}'.format(len(valid_idx), valid_idx[:10]))
    print(len(full_df))
    print(len(full_df.iloc[train_idx]))

    checkpoints_save_name_pref = '{}_{}_'.format(weights_save_path,backbone_name)
    checkpoints_save_name = checkpoints_save_name_pref + '_k_fold_shuffle_ep_'+str(epoch_n) + '_{epoch:03d}_{val_output_grapheme_acc:.4f}.hdf5'
    tensorboard_save_name = '{}logs_{}/'.format(logs_save_path,backbone_name)

    callbacks = [ModelCheckpoint(checkpoints_save_name , 
                    monitor='val_loss', 
                    verbose=1,
                    save_best_only=True),
                # TensorBoard(log_dir=tensorboard_save_name),
                EarlyStopping(monitor='val_loss',
                        patience=15, 
                        verbose=1),
                ReduceLROnPlateau(monitor='val_loss', 
                                factor=0.5,
                                patience=3, 
                                verbose=1)]

# n_fold = 5
# for fold in range(n_fold):
#     train_idx = np.where((full_df['fold'] != fold) & (full_df['unseen'] == 0))[0]
#     valid_idx = np.where((full_df['fold'] == fold) | (full_df['unseen'] != 0))[0]

#     df_this_train = full_df.loc[train_idx].reset_index(drop=True)
#     df_this_valid = full_df.loc[valid_idx].reset_index(drop=True)

    cl_weights = get_class_weights_dict(full_df.iloc[train_idx], columns, output_names)

    train_generator = datagen.flow_from_dataframe(dataframe=full_df.iloc[train_idx],
                                                directory=images_path_train,
                                                color_mode='rgb',
                                                x_col="image_id",
                                                y_col=columns,
                                                batch_size=BATCH_SIZE,
                                                seed=42,
                                                shuffle=True,
                                                class_mode="other",
                                                target_size=(SIZE,SIZE))

    val_generator = datagen.flow_from_dataframe(dataframe=full_df.iloc[valid_idx],
                                                directory=images_path_train,
                                                color_mode='rgb',
                                                x_col="image_id",
                                                y_col=columns,
                                                batch_size=BATCH_SIZE,
                                                seed=42,
                                                shuffle=True,
                                                class_mode="other",
                                                target_size=(SIZE,SIZE))
    print(train_generator.n)
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=val_generator.n//val_generator.batch_size

    history = model.fit_generator(generator=generator_wrapper(train_generator),
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=generator_wrapper(val_generator),
                        validation_steps=STEP_SIZE_VALID,
                        epochs=n_epochs,
                        class_weight=cl_weights,
                        verbose=1,
                        callbacks=callbacks)
    # if epoch == 0:
    #     full_history = history
    # else:
    #     for k in history: full_history[k] = full_history[k] + history[k]

     # Custom ReduceLRonPlateau
    # best_val_loss = CustomReduceLRonPlateau(model, full_history, epoch, best_val_loss)
    del train_generator, val_generator, train_idx, valid_idx
    gc.collect()


# plot_history(full_history, plots_save_path)
	
