
import efficientnet.keras
from keras.models import load_model
import cv2
import glob
import os
import numpy as np
import pandas as pd


model = load_model('weights/best_efficientnetB0.hdf5')

DATASET_PATH = '/home/rauf/datasets/bengali/'
SIZE = 128

columns = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']
images_path_test = os.path.join(DATASET_PATH, 'pngs/','test/')
file_names = glob.glob("{}*.png".format(images_path_test))

row_ids = []
targets = []
for fname in file_names:
    img = cv2.imread(fname, 0)
    img = img/255
    row_id_prefix = fname.split('/')[-1].split('.')[0]
    img = img.reshape(1, *img.shape, 1)
    predictions = model.predict(img)
    pred = {}
    pred['grapheme_root'] = np.argmax(predictions[0], axis=1)[0]
    pred['vowel_diacritic'] = np.argmax(predictions[1], axis=1)[0]
    pred['consonant_diacritic'] = np.argmax(predictions[2], axis=1)[0]
    
    for col in columns:
        row_ids.append(row_id_prefix + '_' + col)
        targets.append(pred[col])
    print(pred)



df_sample = pd.DataFrame(
    {
        'row_id': row_ids,
        'target':targets
    },
    columns = ['row_id','target'] 
)
df_sample.to_csv('submission.csv',index=False)
print(df_sample.head())