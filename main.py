# %%
import warnings

from transformers import logging

from function import ans_len_limit, get_NN, get_NP, transe
from question import *

logging.set_verbosity_warning()
warnings.filterwarnings(action='ignore')
#%%
f = open("/home/qg/testset/1.txt","r")
passageID=1
passage = f.read()
# %%
'''## sh_

a=get_NP(passage)
#%%
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
# print(stop_words)
#%%
a=list(set(a))
#%%
def ans_len_limit(answers):
    for i in answers:
        if len(i.split()) >=3:
            answers.remove(i)
    return answers
aa=a.copy()
ans_len_limit(a)
#%%
def remove_stopwords(stop_words, answers):
    aa=answers.copy()
    for ans in aa:
        if ans  in stop_words:
            aa.remove(ans)
    return aa
wh_answers=remove_stopwords(stop_words, a)'''

#%%
wh=WH()
wh_answers=wh.answers_generate(passage)
wh_question=wh.question_generate(passage, wh_answers)## 정답이 될 수 있는 단어(고유명사?명사구?)를 답으로 가진 질문 list
wh_get_scores=wh.get_scores(wh_question, wh_answers)## 질문과 정답의 적합성 점수 list
#%%
wh_NNs=get_NN(wh_answers)## 명사
#%%
sentences = nltk.sent_tokenize(passage)
#%%
## 정답이 포함된 문장 찾기
def get_sentence_with_ans(sentences, answer):
    target_sentence=[]
    for sentence in sentences:
        if answer in sentence:
            target_sentence.append(sentence)
    ##scientists ’ beliefs 처럼 전처리가 이상하게 된 경우 
    if len(target_sentence)==0:
        return None
    return target_sentence
#%%
wh_target_sentence=[]
#%%
for i in wh_NNs:
    wh_target_sentence.append(get_sentence_with_ans(sentences, i))
#%%
for i in wh_target_sentence:
    if wh_target_sentence[0]=='-1':

#%%
