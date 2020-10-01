import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import cv2
import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
from albumentations import (
    CLAHE, Rotate,
    Blur, GridDistortion, 
    GaussNoise, MotionBlur,  
    RandomBrightnessContrast, OneOf, Compose, Downscale,
    ElasticTransform,
    ShiftScaleRotate
)

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=128, pad=16):
    HEIGHT = 137
    WIDTH = 236
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect rsatio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))

def generate_from_components(g_i, v_i, c_i):
    g_v = graphemes_df.iloc[g_i]['component']
    v_v = vowel_df.iloc[v_i]['component']
    c_v = consonant_df.iloc[c_i]['component']
    vowel_type1 = [2,4,5,6,7,8,9,10]
    vowel_type2 = [1,3]
    consonant_type1 = [1]
    consonant_type2 = [2,3]
    consonant_type3 = [5,4]

    a = 0

    if v_i in vowel_type1:
        if c_i == 0:
            a = g_v + v_v
        elif c_i in consonant_type3:
            a = g_v + c_v  + v_v 
        elif c_i in consonant_type1:
            a = g_v + v_v + c_v
        else:
            a = c_v + g_v + v_v
    elif v_i in vowel_type2:
        if c_i == 0:
            a = g_v + v_v
        else:
            if c_i in consonant_type1:
                a = g_v + v_v+ c_v
            elif c_i in consonant_type2:
                a = c_v + g_v + v_v
            else:
                a = g_v + c_v + v_v
    else:
        if c_i == 0:
            a = g_v
        else:
            if c_i in consonant_type2:
                a = c_v+g_v
            else:
                a = g_v+c_v
    
    return a

def image_from_char(char, font):
    HEIGHT = 137
    WIDTH = 236
    image = Image.new('RGB', (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(image)
    myfont = ImageFont.truetype(font, 50, layout_engine=ImageFont.LAYOUT_RAQM)
    w, h = draw.textsize(char, font=myfont)
    draw.text(((WIDTH - w) / 2,(HEIGHT - h) / 3), char, font=myfont)

    return np.array(image)

df = pd.read_csv('class_map_corrected.csv')
# df = pd.read_csv('class_map.csv')
graphemes_df = df.loc[df['component_type'] == 'grapheme_root'].reset_index()
vowel_df = df.loc[df['component_type'] == 'vowel_diacritic'].reset_index()
consonant_df = df.loc[df['component_type'] == 'consonant_diacritic'].reset_index()


AUGMENTATIONS = Compose([
    ShiftScaleRotate(shift_limit=(-0.01,0.01), scale_limit=(-0.001,0.001), rotate_limit=(-5,5), p=0.8),
    GridDistortion(always_apply=False, p=1.0, num_steps=4, distort_limit=(-0.2, 0.2)),
    ElasticTransform(always_apply=False, p=0.2, alpha=0.3, sigma=10, alpha_affine=5),
])
df_new = pd.DataFrame(columns=list(df_train.columns))
idx = 200840

n_augs = 10
font_1 = 'Nikosh.ttf'
font_2 = 'Siyamrupali_1.ttf'
font_3 = 'kalpurush-2.ttf'
font_4 = 'Bangla.ttf'
fonts = [font_1, font_2, font_3, font_4]
for i in range(168):
    for j in range(11):
        for k in range(7):
            if j==0 and (k==2 or k==3):
                continue
            grapheme_generated = generate_from_components(g_i=i, v_i=j, c_i=k)
            
            for font in fonts:
                image = image_from_char(grapheme_generated, font)
                for aug_i in range(n_augs):
                    data = {"image": image}
                    augmented = AUGMENTATIONS(**data)
                    img = augmented['image']
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img_cropped = crop_resize(img)
#                     img_cropped = 255 - img_cropped
                    img_cropped = (img_cropped*(255.0/img_cropped.max())).astype(np.uint8)
                    cv2.imwrite('images/Train_{}.png'.format(idx), img_cropped)
                    df_new = df_new.append({'image_id': 'Train_{}'.format(idx), 
                                            'grapheme_root': i, 
                                            'vowel_diacritic': j,
                                            'consonant_diacritic': k,
                                            'grapheme': grapheme_generated}, 
                                           ignore_index=True)
                    idx+=1

df_new.to_csv('train_new.csv', sep='\t')
print(df_new)


