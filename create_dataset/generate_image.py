import pandas as pd


def image_from_char(char):
    image = Image.new('RGB', (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(image)
    myfont = ImageFont.truetype('/kaggle/input/kalpurush-fonts/kalpurush-2.ttf', 120)
    w, h = draw.textsize(char, font=myfont)
    draw.text(((WIDTH - w) / 2,(HEIGHT - h) / 3), char, font=myfont)

    return image

df = pd.read_csv('class_map_corrected.csv')
graphemes_df = df.loc[df['component_type'] == 'grapheme_root'].reset_index()
vowel_df = df.loc[df['component_type'] == 'vowel_diacritic'].reset_index()
consonant_df = df.loc[df['component_type'] == 'consonant_diacritic'].reset_index()


graphemes_df.iloc[15]
vowel_df.iloc[9]
consonant_df.iloc[5]
