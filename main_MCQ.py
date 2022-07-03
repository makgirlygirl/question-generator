# %%
# MCQ fin(translate)
import warnings

from transformers import logging

from function import (ans_len_limit, get_NN, get_NP, get_sentence_with_ans,
                      pick_question, transe)
from question import *

logging.set_verbosity_warning()
warnings.filterwarnings(action='ignore')
#%%
f = open("/home/my_qg/testset/2.txt","r")
passageID=2
passage = f.read()
#%%
mcq=MCQ()
mcq_summarize=mcq.summarize(passage)
#%%
mcq_paraphrase=mcq.paraphrase(mcq_summarize)## sent_completion_dict 로 나옴
#%%
mcq_distractors =mcq.distractors(mcq_paraphrase)
#%%
## answer이 여러개가 될 수 있음->list로 나옴
## 화면에 띄워줄때 answer list에서 랜덤으로 하나 뽑아서 문제를 만들거나
## 아니면 그냥 미리 여기서 랜덤으로 고쳐주기(question.py에서 수정)
mcq_dict=mcq.make_dict(passageID, mcq_paraphrase, mcq_distractors)
# %%
