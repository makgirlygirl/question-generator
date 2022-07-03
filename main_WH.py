# %%
# wh fin(translate)
import warnings

from transformers import logging

from function import ans_len_limit, get_NN, get_NP, transe
from question import *

logging.set_verbosity_warning()
warnings.filterwarnings(action='ignore')
#%%
f = open("/home/my_qg/testset/2.txt","r")
passageID=2
passage = f.read()
#%%
wh=WH()
