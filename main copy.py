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

wh=WH()
wh_answers=get_NP(passage)## 고유명사? 명사구?

wh_question=wh.question_generate(passage, wh_answers)## 정답이 될 수 있는 단어(고유명사?명사구?)를 답으로 가진 질문 list
wh_get_scores=wh.get_scores(wh_question, wh_answers)## 질문과 정답의 적합성 점수 list
    

scores=wh_get_scores
answers=wh_answers
    ## 정답의 단어 개수 len() <= 4 사용한다. 
for i in range(len(scores)):
    index = scores[i]
    if len(answers[index].split(' ')) > 4:
       continue
    break
    
sentences = nltk.sent_tokenize(passage)
for sentence in sentences:
    if answers[index] in sentence:
        target_sentence = sentence

NNs = get_NN(answers[index])

distractors = []
for i in range(3):
    distractors.append(d_generator.generate_distractor(target_sentence, 9-i, answers[index], NNs))
