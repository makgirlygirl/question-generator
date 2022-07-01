# %%
import warnings

from transformers import logging

from function import get_NN, get_NP
from question import *

logging.set_verbosity_warning()
warnings.filterwarnings(action='ignore')
#%%
f = open("/home/qg/testset/1.txt","r")
passageID=1
passage = f.read()
#%%
mcq=MCQ()
mcq_summarize=mcq.summarize(passage)
mcq_paraphrase=mcq.paraphrase(mcq_summarize)## sent_completion_dict 로 나옴
mcq_distractors = mcq.distractors(mcq_paraphrase)
mcq_dict=mcq.make_dict(passageID, mcq_paraphrase, mcq_distractors)
# %%
wh=WH()
wh_answers=get_NP(passage)## 고유명사? 명사구?
wh_question=wh.question_generate(passage, wh_answers)## 정답이 될 수 있는 단어(고유명사?명사구?)를 답으로 가진 질문 list
wh_get_scores=wh.get_scores(wh_question, wh_answers)## 질문과 정답의 적합성 점수 list
#%%
wh_NNs=get_NN(wh_answers)## 명사
#%%
# wh_get_NN=wh=wh.get_NN(passage, candidate, wh_answers, wh_NNs) ## 오답 list
# wh_make_dict=wh.make_dict(passageID, wh_get_NN)

#%%
# %%

mcq_dict=mcq.make_dict(passageID, mcq_sent_completion_dict, mcq_false_sentence)

# %%
# %%
