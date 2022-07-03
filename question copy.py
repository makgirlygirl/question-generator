#%%
######## translate 하기 전
from transformers import logging

logging.set_verbosity_warning()
import warnings

import nltk
import torch

warnings.filterwarnings(action='ignore')
# nltk.download('stopwords')
from nltk.corpus import stopwords

from function import (ans_len_limit, benepar_parser, encode_qa_pairs,
                      generate_distractor, generate_sentences, get_flattened,
                      get_NN, get_NP, get_sentence_completions, preprocess,
                      remove_stopwords, sent_tokenize)
from model import (bert_model, gpt2_model, gpt2_tokenizer, paraphrase_model,
                   paraphrase_tokenizer, qae_model, qae_tokenizer, qg_model,
                   qg_tokenizer, summarize_model, summarize_tokenizer,
                   unmasker)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%
question_dict={'passageID':None,
                'question_type':None,
                'question':None, 
                'answer':None,
                'd1':None, 'd2':None, 'd3':None, 'd4':None}
#%%
## MCQ 쓸 유형별로 조금씩 변경해서 MCQ1, MCQ2, ...등 만들기->question_type에 이걸 넣을것
class MCQ:
    def __init__(self):
        self.summarize_tokenizer=summarize_tokenizer
        self.paraphrase_tokenizer = paraphrase_tokenizer
        self.gpt2_tokenizer = gpt2_tokenizer

        self.summarize_model = summarize_model
        self.paraphrase_model = paraphrase_model
        self.gpt2_model = gpt2_model
        self.bert_model=bert_model
    
    # def summarize(self, passageID):
        # passageID로 passage를 가져온다(db랑 연결이 되어야 할 듯 ??)
    # 일단은 passage로 함수를 만들어봄
    def summarize(self, passage):
        ## passageID로 passage를 가져온다(db랑 연결이 되어야 할 듯 ??)
        inputs = summarize_tokenizer.encode("summarize: " + passage, return_tensors="pt", max_length=2000)
        inputs = inputs.to(DEVICE)
        outputs = summarize_model.generate(inputs, max_length=1000, min_length=100, length_penalty=2.0, num_beams=4, early_stopping=True)

        extracted_sentences = summarize_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        tokenized_sentences = nltk.tokenize.sent_tokenize(extracted_sentences)

        filter_quotes_and_questions = preprocess(tokenized_sentences)
        ## 5문장을 리턴할 수는 없을까?
        return filter_quotes_and_questions# list
    
    ## summarize의 결과가 paraphrase의 input으로 들어감
    def paraphrase(self, filter_quotes_and_questions):# list->list
        paraphrased_sentences=[]
        for summary_idx in range(len(filter_quotes_and_questions)):
            sentence = filter_quotes_and_questions[summary_idx]
            inputs = "paraphrase: " + sentence + " </s>"

            encoding = paraphrase_tokenizer.encode_plus(inputs, pad_to_max_length=True, return_tensors="pt")
            input_ids, attention_masks = encoding["input_ids"].to(DEVICE), encoding["attention_mask"].to(DEVICE)

            outputs = paraphrase_model.generate(
                input_ids=input_ids, attention_mask=attention_masks,
                max_length=256,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                early_stopping=True,
                num_return_sequences=4
                )

            paraphrased_sentences.append(paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True,clean_up_tokenization_spaces=True))
        
            if len(paraphrased_sentences) == 4:
                break

            if (summary_idx == (len(filter_quotes_and_questions) - 1)) & (len(paraphrased_sentences) < 4): # 마지막인데 채워지지 않았을 경우 존재하는 paraphrased sentence 반복해서 false 문장 생성

                for paraphrase_idx in range(1, 5):
                    paraphrased_sentences.append(paraphrase_tokenizer.decode(outputs[paraphrase_idx], skip_special_tokens=True,clean_up_tokenization_spaces=True))
        sent_completion_dict=get_sentence_completions(paraphrased_sentences)
        # print(paraphrased_sentences)
        # print(sent_completion_dict)
        return sent_completion_dict

    ## paraphrase 의 결과가  complete_dict의 input으로 들어감
    def distractors(self, sent_completion_dict):
        distractors=[]
        distractor_cnt = 1

        for key_sentence in sent_completion_dict:
            if distractor_cnt == 5:
                break
            partial_sentences = sent_completion_dict[key_sentence]
            false_sentences =[]
            # df_TFQuestions.loc[0, 'id'] = 
            # print_string = "**%s) True Sentence (from the story) :**"%(str(index))
            # printmd(print_string)
            # print ("  ",key_sentence)
            false_sents = []
            for partial_sent in partial_sentences:
                for repeat in range(10):
                    false_sents = generate_sentences(partial_sent, key_sentence)
                    if false_sents != []:
                        break
                        
                false_sentences.extend(false_sents)
            distractors.append(false_sentences[0])
            distractor_cnt += 1
        return distractors

    ## paraphrase 의 결과가  complete_dict의 input으로 들어감
    def make_dict(self, passageID, sent_completion_dict,false_sentences ):
        question_dict['passageID']=int(passageID)
        question_dict['question_type']='MCQ'## 문제 유형에 따라 MCQ1, MCQ2, ...
        # question_dict['question'] = '다음 중 주제로 적절한 것은?' ## 문제 유형에 따라 매핑
        question_dict['answer']=list(sent_completion_dict.keys())[0]## 왜 0이지??
        question_dict['d1']=false_sentences[0]
        question_dict['d2']=false_sentences[1]
        question_dict['d3']=false_sentences[2]
        question_dict['d4']=false_sentences[3]

        return question_dict

# # %%
class WH:
    def __init__(self):
        self.qg_tokenizer=qg_tokenizer
        self.qae_tokenizer = qae_tokenizer

        self.qg_model = qg_model
        self.qae_model = qae_model

        self.unmasker = unmasker

    ## 정답 생성
    def answers_generate(self, passage):
        answers=get_NP(passage)
        answers=ans_len_limit(answers)
        stop_words = set(stopwords.words('english'))
        answers=remove_stopwords(stop_words, answers)
        return answers
    ## get_NP의 결과가 question_generate의 인풋으로 들어감
    ## 문제 생성
    def question_generate(self, passage, answers):
        ANSWER_TOKEN = "<answer>"
        CONTEXT_TOKEN = "<context>"
        SEQ_LENGTH = 512## 2000이었나? 아무튼 바꾸기(function.py의 encode_qa_pairs 에도 있음)
        
        questions = []

        for ans in answers: 
            qg_input = "{} {} {} {}".format(ANSWER_TOKEN, ans, CONTEXT_TOKEN, passage)
                
            encoded_input =qg_tokenizer(qg_input, padding='max_length', max_length=SEQ_LENGTH, truncation=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                output =qg_model.generate(input_ids=encoded_input["input_ids"])
            question = qg_tokenizer.decode(output[0], skip_special_tokens=True)
            questions.append(question)
        return questions

    ## question_generate, get_NP의 결과가 인풋으로 들어감
    ## 문제 평가
    def get_scores(self, questions, answers):
        encoded_qa_pairs=encode_qa_pairs(questions, answers)## list: 인코딩
        scores = {}
        qae_model.eval()
        with torch.no_grad():
            for i in range(len(encoded_qa_pairs)):
                scores[i] = self.qae_model(**encoded_qa_pairs[i])[0][0][1]
        return [k for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)]

    
    ## 오답생성
    def distractors(self, passage, answers, NN):
        NN=get_NN(answers)
        sentences = nltk.sent_tokenize(passage)
        for sentence in sentences:
            if answers in sentence:
                target_sentence = sentence
        
        distractors = []
        for i in range(4):
            distractors.append(generate_distractor(target_sentence, 9-i, answers, NN))
    
    ## dict 리턴
    def make_dict(self, passageID, answers, question, distractors):
        question_dict['passageID']=int(passageID)
        question_dict['question_type']='WH'
        question_dict['question'] = question.split("?")[0]+"?"
        # question_dict['answer']=answers
        # question_dict['d1']=distractors[0]
        # question_dict['d2']=distractors[1]
        # question_dict['d3']=distractors[2]
        # question_dict['d4']=distractors[3]

        # distractors = []
        # for i in range(3):
        #     distractors.append(generate_distractor(target_sentence, 9-i, answers[index], NNs))
    

        return question_dict


# %%
