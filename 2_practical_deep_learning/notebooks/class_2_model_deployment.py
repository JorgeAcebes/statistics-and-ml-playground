# %% [markdown]
# ## Alligator or Crocodile?
# %%
import os
import shutil
import time
import ssl
import torch
import random
import kagglehub
import warnings
import tarfile
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


# Fastcore patch must be applied before importing fastai to guarantee inheritance
import fastcore.foundation
def _starmap(self, f, *args, **kwargs):
    return self.map(lambda o: f(*o, *args, **kwargs))
fastcore.foundation.L.starmap = _starmap

from fastcore.all import *
from fastai.vision.all import *

print('Imported libraries')

# %%
SCRIPT_DIR = Path(__file__).resolve().parent
PRACTICAL_DL_DIR = SCRIPT_DIR.parent
DATA_DIR = PRACTICAL_DL_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True, parents=True)
DOCS_DIR = PRACTICAL_DL_DIR / 'docs'
DOCS_DIR.mkdir(exist_ok=True, parents=True)
path = DATA_DIR / 'alig_or_croc'

# %% [markdown]
# ## Step 1: Download images from Kaggle

# %%
print("Downloading Kaggle datasets...")

# Download latest version
path_alig_croc = Path(kagglehub.dataset_download("azharn/alligator-vs-crocodile1"))

print("Path to dataset files:", path_alig_croc)

# %%
alig_dest = path / 'alig'
croc_dest = path / 'croc'

# Deleting previous executions 
if alig_dest.exists(): shutil.rmtree(alig_dest)
if croc_dest.exists(): shutil.rmtree(croc_dest)

alig_dest.mkdir(exist_ok=True, parents=True)
croc_dest.mkdir(exist_ok=True, parents=True)


alig_dirs = [p for p in path_alig_croc.rglob("alligator") if p.is_dir()]
croc_dirs = [p for p in path_alig_croc.rglob("crocodile") if p.is_dir()]


all_alig_files = get_image_files(alig_dirs[0])
all_croc_files = get_image_files(croc_dirs[0])


def sample_and_copy(src_files, dest_dir):
    n = len(src_files)
    sample = random.sample(list(src_files), n)
    for i, f in enumerate(sample):
        shutil.copy(f, dest_dir / f"{dest_dir.name}_{i}{f.suffix}")

sample_and_copy(all_alig_files, alig_dest)
sample_and_copy(all_croc_files, croc_dest)

print("Resizing images (this might take a while)...")
resize_images(alig_dest, max_workers=0, max_size=300, dest=alig_dest)
resize_images(croc_dest, max_workers=0, max_size=300, dest=croc_dest)
print("Resizing completed")

# Reference image for the last test
dest_alig = get_image_files(alig_dest)[0]

# %%

img = PILImage.create(dest_alig)
show_image(img, title=dest_alig.stem)


# %% [markdown]
# ## Step 2: Clean and Balance Data

# %%
warnings.filterwarnings("ignore", category=UserWarning)

files = get_image_files(path)
failed = []

print("Checking tensors' integrity...")
for fn in files:
    try:
        with Image.open(fn) as img:
            img.load() 
    except Exception:
        failed.append(fn)

failed = L(failed)
failed.map(Path.unlink)
print(f'Corrupted images removed: {len(failed)}')

# %% [markdown]
# ## Step 3: Train our model

# %%

# Let's see how changing the resize method affects the loss
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='crop')]
).dataloaders([alig_dest, croc_dest], bs=32)

dls.show_batch(max_n=6)

dls_squish = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders([alig_dest, croc_dest], bs=32)

dls_squish.show_batch(max_n=6)

# %%

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)

learn_squish = vision_learner(dls_squish, resnet18, metrics=error_rate)
learn_squish.fine_tune(3)


# %% [markdown]
# ## Step 4: Use our model

# %%
is_alig, _, probs = learn.predict(PILImage.create(dest_alig))
print(f"This is a: {is_alig}.")
print(f"Probability it's an alligator: {probs[0]:.4f}")

# %%



def alig_croc_pred(dest):
    is_alig, _, probs = learn.predict(PILImage.create(dest))
    print('='*50)
    print(f"Method of resizing: CROP\nAI says this is a: {is_alig}. Image name: {dest.stem}")
    print(f"Probability it's an alligator: {probs[0]:.4f}")
    print('='*50)


def alig_croc_pred_squish(dest):
    is_alig, _, probs = learn_squish.predict(PILImage.create(dest))
    print('='*50)
    print(f"Method of resizing: SQUISH\nAI says this is a: {is_alig}. Image name: {dest.stem}")
    print(f"Probability it's an alligator: {probs[0]:.4f}")
    print('='*50)

# %%

pred = [p for p in path.iterdir() 
        if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]


for p in pred: 
    alig_croc_pred(p)
    alig_croc_pred_squish(p)

# %% [markdown]
#  ## Step 5: Export our model

learn.export('model.pkl')
