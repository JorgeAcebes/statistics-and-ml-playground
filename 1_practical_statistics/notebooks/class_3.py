#%%
# %% [markdown]
# ## In this class...
# - lista
# - otra cosa
#%%
# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import statistics
from scipy.stats import norm

from pathlib import Path
print('Imported libraries')

# %%
notebook = Path.cwd().parent
data_path= notebook/'data'

demo = pd.read_csv(data_path/'demography.csv', sep=';')
demo

# %%