# %% [markdown]
# ## Is it a bird?
# %% [markdown]

# Original code [here](https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data).

# I recommend working with the other script (class_1_is_it_a_bird_with_kaggle.py)

# %%
import os
import sys
import time
import ssl
import torch
from pathlib import Path


# Fastcore patch must be applied before importing fastai to guarantee inheritance
import fastcore.foundation
def _starmap(self, f, *args, **kwargs):
    return self.map(lambda o: f(*o, *args, **kwargs))
fastcore.foundation.L.starmap = _starmap

from fastcore.all import *
from fastai.vision.all import *
from fastdownload import download_url
from ddgs import DDGS 

print('Imported libraries')

# %%


SCRIPT_DIR = Path(__file__).resolve().parent
PRACTICAL_DL_DIR = SCRIPT_DIR.parent
DATA_DIR = PRACTICAL_DL_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True, parents=True)



# %% [markdown]
# ## Step 1: Download images of birds and non-birds

# %%
def search_images(keywords, max_images=200): 
    time.sleep(2)
    return L(DDGS().images(keywords, max_results=max_images)).itemgot('image')

# %% [markdown]
# Let's start by searching for a bird photo and seeing what kind of result we get. We'll start by getting URLs from a search:

# %%
urls = search_images('bird photos', max_images=1)
urls[0]

# %% [markdown]
# ...and then download a URL and take a look at it:

# %%
ssl._create_default_https_context = ssl._create_unverified_context

dest_bird = DATA_DIR / 'bird.jpg'
download_url(urls[0], dest_bird, show_progress=False)

im = Image.open(dest_bird)
im.to_thumb(256,256)

# %% [markdown]
# Now let's do the same with "forest photos":

# %%
dest_forest = DATA_DIR / 'forest.jpg'
download_url(search_images('forest photos', max_images=1)[0], dest_forest, show_progress=False)
Image.open(dest_forest).to_thumb(256,256)

# %% [markdown]
# Our searches seem to be giving reasonable results, so let's grab a few examples of each of "bird" and "forest" photos, and save each group of photos to a different folder:

# %%
searches = 'forest', 'bird'
path = Path(DATA_DIR / 'bird_or_not')

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    time.sleep(10)
    resize_images(path/o, max_workers=1, max_size=300, dest=path/o)

# %% [markdown]
# ## Step 2: Clean and Balance Data

# %%
from PIL import Image
import warnings

# 1. Verify and remove corrupted images sequentially to avoid BrokenProcessPool on Windows
# We ignore harmless ICC profile warnings from PIL
warnings.filterwarnings("ignore", category=UserWarning)

files = get_image_files(path)
failed = []

# Iterate through each file and force a full memory load to catch C-level segfaults
for fn in files:
    try:
        with Image.open(fn) as img:
            img.load() 
    except Exception:
        # If the image is truncated or completely corrupted, append it to the fail list
        failed.append(fn)

failed = L(failed)
failed.map(Path.unlink)
print(f'Corrupted images removed: {len(failed)}')

# 2. Balance the dataset sizes SECOND (only after removing corrupted ones)
bird = path / 'bird'
fore = path / 'forest'

files_bird = get_image_files(bird).sorted()
files_fore = get_image_files(fore).sorted()

tot_bird = len(files_bird)
tot_fore = len(files_fore)

print(f'Initial count -> Birds: {tot_bird} | Forest: {tot_fore}')

# Truncate the larger class to match the size of the smaller class
if tot_fore > tot_bird:
    to_remove = files_fore[tot_bird:]
    for f in to_remove: f.unlink()
elif tot_fore < tot_bird:
    to_remove = files_bird[tot_fore:]
    for f in to_remove: f.unlink()

print(f'Final count -> Birds: {len(get_image_files(bird))} | Forest: {len(get_image_files(fore))}')

# %% [markdown]
# ## Step 3: Train our model

# %%
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=8)

dls.show_batch(max_n=6)


# Some bird photos are AI slop :(

# %% [markdown]
# Here what each of the `DataBlock` parameters means:
# 
#     blocks=(ImageBlock, CategoryBlock),
# 
# The inputs to our model are images, and the outputs are categories (in this case, "bird" or "forest").
# 
#     get_items=get_image_files, 
# 
# To find all the inputs to our model, run the `get_image_files` function (which returns a list of all image files in a path).
# 
#     splitter=RandomSplitter(valid_pct=0.2, seed=42),
# 
# Split the data into training and validation sets randomly, using 20% of the data for the validation set.
# 
#     get_y=parent_label,
# 
# The labels (`y` values) is the name of the `parent` of each file (i.e. the name of the folder they're in, which will be *bird* or *forest*).
# 
#     item_tfms=[Resize(192, method='squish')]
# 
# Before training, resize each image to 192x192 pixels by "squishing" it (as opposed to cropping it).

# %% [markdown]
# Now we're ready to train our model. The fastest widely used computer vision model is `resnet18`.

# %%
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)

# %% [markdown]
# ## Step 4: Use our model

# %%
# Using the correctly scoped variable from earlier
is_bird, _, probs = learn.predict(PILImage.create(dest_bird))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}")
# %%

# I will also try with some photos I took in my trip to Costa Rica

bird_CR = DATA_DIR/'colibri.jpg'
forest_CR = DATA_DIR /'forest_CR.jpg'
guacamayo_CR = DATA_DIR/'guacamayo.jpg'
butterfly_CR = DATA_DIR/'butterfly.jpg'

to_check = [bird_CR, forest_CR, guacamayo_CR, butterfly_CR]


def bird_checker(path: Path):
    is_bird, _, probs = learn.predict(PILImage.create(path))
    print(f"AI says this is a: {is_bird}. Original name: {path.stem}")
    print(f"Probability it's a bird: {probs[0]:.4f}")


for check in to_check:
    bird_checker(check)
