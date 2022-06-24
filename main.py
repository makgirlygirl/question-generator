# %%
from question import *

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
# wh=WH()
# ## 명사구 말고도 다른거 ,,,?
# wh_answer=wh.get_NP()## 정답으로 사용될 명사구에 해당하는 단어
# wh_generate_question=wh.generate_question()
# wh_QAevaluator=wh.QAevaluator()
