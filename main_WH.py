# %%
# wh fin(translate)
import warnings

from transformers import logging

from function import get_sentence_with_ans, pick_question, transe
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
def generate_distractor(self, text, candidate, answers, NNs: list):
    distractor = []
    divided = word_tokenize(text)
    substitute_word = NNs[0]

    mask_index = divided.index(substitute_word)
    divided.pop(mask_index)

    divided.insert(mask_index, '[MASK]')
    text = ' '.join(divided)
    unmasked_result = self.unmasker(text, top_k=10)[candidate]

    text = unmasked_result["sequence"]

    answers = answers.split(' ')
    answer_index = answers.index(substitute_word)
    answers.pop(answer_index)
    answers.insert(answer_index, unmasked_result["token_str"])
    return " ".join(answers)
#%%
wh_distractors=[]
for i in wh_answers:
    wh_distractors.append(generate_distractor(passage, i))
#%%
#%%
# wh_target_sentence=[]
# wh_NNs=[]
# distractors = []

# for i in wh_answers:
#     # wh_NNs.append(get_NN(i))
#     # wh_target_sentence.append(get_sentence_with_ans(sentences, i))
#     wh_NNs=get_NN(i)
#     wh_target_sentence=get_sentence_with_ans(sentences, i)
#     for j in range(4):
#         distractors=wh.generate_distractor(wh_target_sentence, 9-j, i, wh_NNs)
#     print(distractors)
# %%
