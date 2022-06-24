#%%
import nltk
import torch

from function import generate_sentences, get_sentence_completions, preprocess
from model import (bert_model, gpt2_model, gpt2_tokenizer, paraphrase_model,
                   paraphrase_tokenizer, summarize_model, summarize_tokenizer)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%
question_dict={'passageID':None,
                'question_type':None,
                'question':None, 
                'answer':None,
                'd1':None, 'd2':None, 'd3':None, 'd4':None}## distractors list?? 아니면 d1, d2, d3, d4??
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
        inputs = summarize_tokenizer.encode("summarize: " + passage, return_tensors="pt", max_length=512)
        inputs = inputs.to(DEVICE)
        outputs = summarize_model.generate(inputs, max_length=300, min_length=100, length_penalty=2.0, num_beams=4, early_stopping=True)

        extracted_sentences = summarize_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        tokenized_sentences = nltk.tokenize.sent_tokenize(extracted_sentences)

        filter_quotes_and_questions = preprocess(tokenized_sentences)
        
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
                num_return_sequences=3
                )

            paraphrased_sentences.append(paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True,clean_up_tokenization_spaces=True))
        
            if len(paraphrased_sentences) == 4:## 5지선다니까 4개
                # print('4 filled')
                break

            if (summary_idx == (len(filter_quotes_and_questions) - 1)) & (len(paraphrased_sentences) < 3): # 마지막인데 채워지지 않았을 경우 존재하는 paraphrased sentence 반복해서 false 문장 생성
                # print(summary_idx)
                # print('hit')
                for paraphrase_idx in range(1, 3):
                    paraphrased_sentences.append(paraphrase_tokenizer.decode(outputs[paraphrase_idx], skip_special_tokens=True,clean_up_tokenization_spaces=True))

        return paraphrased_sentences

    ## paraphrase 의 결과가  complete_dict의 input으로 들어감
    def make_dict(self, passageID, paraphrased_sentences):
        sent_completion_dict = get_sentence_completions(paraphrased_sentences)
        question_dict['passageID']=int(passageID)
        question_dict['question_type']='MCQ'## 문제 유형에 따라 MCQ1, MCQ2, ...
        # question_dict['question'] = '다음 중 주제로 적절한 것은?' ## 문제 유형에 따라 매핑
        question_dict['answer']=list(sent_completion_dict.keys())[0]

        # question_table[d1]~question_table[d4]
        distractor_cnt = 1

        for key_sentence in sent_completion_dict:

            if distractor_cnt == 5:
                break

            partial_sentences = sent_completion_dict[key_sentence]
            false_sentences =[]
            
            false_sents = []
            for partial_sent in partial_sentences:
                
                for repeat in range(10):
                    false_sents = generate_sentences(partial_sent, key_sentence)
                    if false_sents != []:
                        break
                        
                false_sentences.extend(false_sents)
            
            question_dict[f'd{distractor_cnt}'] = false_sentences[0]
            distractor_cnt += 1

        return question_dict

# # %%
# class WH:
#     def __init__(self):

# %%
