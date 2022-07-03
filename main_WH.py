# %%
# wh fin(translate)
import warnings

from transformers import logging

from function import pick_question, transe
from question import *

logging.set_verbosity_warning()
warnings.filterwarnings(action='ignore')
#%%
f = open("/home/my_qg/testset/2.txt","r")
passageID=2
passage = f.read()
#%%
wh=WH()
wh_answers=wh.answers_generate(passage)
wh_question=wh.question_generate(passage, wh_answers)
wh_get_scores=wh.get_scores(wh_question, wh_answers)

wh_answers, wh_question, wh_get_scores=pick_question(wh_answers, wh_question, wh_get_scores)
#%%

'''
## model
from keybert import KeyBERT

kw_model = KeyBERT()
#%%
def get_keyword(passage):
    result=[]
    keywords = kw_model.extract_keywords(passage, keyphrase_ngram_range=(1, 1), top_n=50, stop_words='english')

    for kw in keywords:
    #     if kw[1]>0.5:
            result.append(kw[0])
    return result
import itertools
#%%
from difflib import SequenceMatcher


def word_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def answers_generate(passage):
        answers=get_keyword(passage)
        print(answers)
        pair=list(itertools.combinations((answers),2))
        for i in pair:
            a=i[0]; b=i[1]
            if word_similarity(a, b)>0.8:
                answers.remove(b)
        return answers
# wh_answers=answers_generate(passage)

def pick_question(answers, question, score):
    answers_r=[];questions_r=[];scores_r=[]
    for i in range(len(score)):
        if score[i]>30:
            print(score[i])
            answers_r.append(answers[i])
            questions_r.append(question[i])
            scores_r.append(score[i])
    return answers_r, questions_r, scores_r
# print(pick_question(wh_answers, wh_question, wh_get_scores))
wh_answers, wh_question, wh_get_scores=pick_question(wh_answers, wh_question, wh_get_scores)
'''
# %%
