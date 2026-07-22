# %% [markdown]
# ## Is it a bird? (Kaggle Version)
# %%
import os
import shutil
import time
import ssl
import torch
import random
import kagglehub
import warnings
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
path = DATA_DIR / 'bird_or_not'

# %% [markdown]
# ## Step 1: Download images from Kaggle

# %%
print("Downloading Kaggle datasets...")

# BIRDS
path_bird_ds = Path(kagglehub.dataset_download("veeralakrishna/200-bird-species-with-11788-images"))
import tarfile
tgz_files = list(Path(path_bird_ds).rglob("CUB_200_2011.tgz")) # Searching for the tgz file

if tgz_files:
    tgz_path = tgz_files[0]

    # Extracting the photos from the .tgz
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=tgz_path.parent, filter='data')


# FOREST
path_forest_ds = Path(kagglehub.dataset_download("rahmasleam/intel-image-dataset"))
# Finding just the forest images:
forest_dirs = [p for p in path_forest_ds.rglob("forest") if p.is_dir()]


# %%
bird_dest = path / 'bird'
fore_dest = path / 'forest'

# Deleting previous executions 
if bird_dest.exists(): shutil.rmtree(bird_dest)
if fore_dest.exists(): shutil.rmtree(fore_dest)

bird_dest.mkdir(exist_ok=True, parents=True)
fore_dest.mkdir(exist_ok=True, parents=True)

all_forest_files = get_image_files(forest_dirs[0])

all_bird_files = get_image_files(path_bird_ds)

N_f = len(all_forest_files)
N_b = len(all_bird_files)
N_max = min(N_f, N_b)

print(f"Total photos in Kaggle -> Forest: {N_f} | Birds: {N_b}")

def sample_and_copy(src_files, dest_dir, n):
    sample = random.sample(list(src_files), n)
    for i, f in enumerate(sample):
        shutil.copy(f, dest_dir / f"{dest_dir.name}_{i}{f.suffix}")

sample_and_copy(all_forest_files, fore_dest, n=N_max)
sample_and_copy(all_bird_files, bird_dest, n=N_max)

print("Resizing images (this might take a while)...")
resize_images(bird_dest, max_workers=0, max_size=300, dest=bird_dest)
resize_images(fore_dest, max_workers=0, max_size=300, dest=fore_dest)

# Reference image for the last test
dest_bird = get_image_files(bird_dest)[0]

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

files_bird = get_image_files(bird_dest).sorted()
files_fore = get_image_files(fore_dest).sorted()

tot_bird = len(files_bird)
tot_fore = len(files_fore)

if tot_fore > tot_bird:
    to_remove = files_fore[tot_bird:]
    for f in to_remove: f.unlink()
elif tot_fore < tot_bird:
    to_remove = files_bird[tot_fore:]
    for f in to_remove: f.unlink()

print(f'Dataset ready for training -> Birds: {len(get_image_files(bird_dest))} | Forest: {len(get_image_files(fore_dest))}')

# %% [markdown]
# ## Step 3: Train our model

# %%
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)

# %%
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)

# %% [markdown]
# ## Step 4: Use our model

# %%
is_bird, _, probs = learn.predict(PILImage.create(dest_bird))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}")

# %%
# Validation with photos from my trip to Costa Rica (there are some tricky ones)
bird_CR = DATA_DIR / 'colibri.jpg'
forest_CR = DATA_DIR / 'forest_CR.jpg'
guacamayo_CR = DATA_DIR / 'guacamayo.jpg'
butterfly_CR = DATA_DIR / 'butterfly.jpg'
sea_bird_CR = DATA_DIR /'sea_bird.jpg'
giant_wasp_CR = DATA_DIR/'giant_wasp.jpg'

to_check = [bird_CR, forest_CR, guacamayo_CR, butterfly_CR, sea_bird_CR, giant_wasp_CR]

def bird_checker(file_path: Path):
    if file_path.exists():
        is_bird, _, probs = learn.predict(PILImage.create(file_path))
        print(f"AI says this is a: {is_bird}. Original name: {file_path.stem}")
        print(f"Probability it's a bird: {probs[0]:.4f}")
    else:
        print(f"File not found: {file_path}")

for check in to_check:
    bird_checker(check)