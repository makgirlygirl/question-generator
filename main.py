# %%
import warnings

from question import *

warnings.filterwarnings(action='ignore')
#%%
f = open("/home/qg/testset/1.txt","r")
passageID=1
passage = f.read()
#%%
mcq=MCQ()
mcq_summarize=mcq.summarize(passage)
mcq_paraphrase=mcq.paraphrase(mcq_summarize)
mcq_make_dict=mcq.make_dict(passageID, mcq_paraphrase)
# %%
wh=WH()
wh_get_NP=wh.get_NP(passage)
wh_question_generate=wh.question_generate(passage, wh_get_NP)
wh_get_scores=wh.get_scores(wh_question_generate, wh_get_NP)
#%%
wh_get_NN=wh=wh.get_NN(passage, candidate, wh_get_NP, NNs) ## 오답 list
wh_make_dict=wh.make_dict(passageID, wh_get_NN)
