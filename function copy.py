#%% package import
##### translate 하기전
from transformers import logging

logging.set_verbosity_warning()
import re
import warnings
from string import punctuation

from nltk.corpus import stopwords

warnings.filterwarnings(action='ignore')
import benepar
import nltk
import scipy
import spacy
import torch
from nltk import sent_tokenize, tokenize, word_tokenize

from model import (bert_model, gpt2_model, gpt2_tokenizer, qae_model,
                   qae_tokenizer, qg_model, qg_tokenizer, translator, unmasker)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)

warnings.filterwarnings(action='ignore')
nlp = spacy.load('en_core_web_sm')

nltk.download('punkt')

benepar.download('benepar_en3')
benepar_parser = benepar.Parser("benepar_en3")

#%% MCQ
def preprocess(sentences):
    output = []
    for sent in sentences:
        single_quotes_present = len(re.findall(r"['][\w\s.:;,!?\\-]+[']",sent))>0
        double_quotes_present = len(re.findall(r'["][\w\s.:;,!?\\-]+["]',sent))>0
        question_present = "?" in sent
        if single_quotes_present or double_quotes_present or question_present :
            continue
        else:
            output.append(sent.strip(punctuation))
    return output
# %% MCQ
def get_flattened(t):# MCQ, WH
    sent_str_final = None
    if t is not None:
        sent_str = [" ".join(x.leaves()) for x in list(t)]
        sent_str_final = [" ".join(sent_str)]
        sent_str_final = sent_str_final[0]
    return sent_str_final

def get_termination_portion(main_string,sub_string):
    combined_sub_string = sub_string.replace(" ","")
    main_string_list = main_string.split()
    last_index = len(main_string_list)
    for i in range(last_index):
        check_string_list = main_string_list[i:]
        check_string = "".join(check_string_list)
        check_string = check_string.replace(" ","")
        if check_string == combined_sub_string:
            return " ".join(main_string_list[:i])       
    return None

def get_right_most_VP_or_NP(parse_tree,last_NP = None,last_VP = None):
    if len(parse_tree.leaves()) == 1:
        return get_flattened(last_NP),get_flattened(last_VP)
    last_subtree = parse_tree[-1]
    if last_subtree.label() == "NP":
        last_NP = last_subtree
    elif last_subtree.label() == "VP":
        last_VP = last_subtree
    
    return get_right_most_VP_or_NP(last_subtree,last_NP,last_VP)

def get_sentence_completions(key_sentences):
    sentence_completion_dict = {}
    for individual_sentence in key_sentences:
        sentence = individual_sentence.rstrip('?:!.,;')
        tree = benepar_parser.parse(sentence)
        last_nounphrase, last_verbphrase =  get_right_most_VP_or_NP(tree)
        phrases= []
        if last_verbphrase is not None:
            verbphrase_string = get_termination_portion(sentence,last_verbphrase)
            if verbphrase_string is not None:
                phrases.append(verbphrase_string)
                
        if last_nounphrase is not None:
            nounphrase_string = get_termination_portion(sentence,last_nounphrase)
            if nounphrase_string is not None:
                phrases.append(nounphrase_string)
    
        longest_phrase =  sorted(phrases, key=len, reverse=True)
        if len(longest_phrase) == 2:
            first_sent_len = len(longest_phrase[0].split())
            second_sentence_len = len(longest_phrase[1].split())
            if (first_sent_len - second_sentence_len) > 4:
                del longest_phrase[1]
                
        if len(longest_phrase)>0:
            sentence_completion_dict[sentence]=longest_phrase

    return sentence_completion_dict
# %% MCQ
def sort_by_similarity(original_sentence, generated_sentences_list):

    sentence_embeddings = bert_model.encode(generated_sentences_list)
    queries = [original_sentence]
    query_embeddings = bert_model.encode(queries)
    number_top_matches = len(generated_sentences_list)
    dissimilar_sentences = []

    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        for idx, distance in reversed(results[0:number_top_matches]):
            score = 1-distance
            # print(score)
            if score < 0.99:
                dissimilar_sentences.append(generated_sentences_list[idx].strip())
           
    sorted_dissimilar_sentences = sorted(dissimilar_sentences, key=len)
    # print('sorted_dissimilar_sentences\n\n')
    # print(sorted_dissimilar_sentences)
    return sorted_dissimilar_sentences[:2]

def generate_sentences(partial_sentence,full_sentence):
    input_ids = gpt2_tokenizer.encode(partial_sentence, return_tensors='pt') # use tokenizer to encode
    input_ids = input_ids.to(DEVICE)
    maximum_length = len(partial_sentence.split())+80 

    sample_outputs = gpt2_model.generate( 
        input_ids,
        do_sample=True,
        max_length=maximum_length, 
        top_p=0.90, 
        top_k=50,   
        repetition_penalty  = 10.0,
        num_return_sequences=5
    )
    generated_sentences=[]
    for i, sample_output in enumerate(sample_outputs):
        decoded_sentences = gpt2_tokenizer.decode(sample_output, skip_special_tokens=True)
        decoded_sentences_list =tokenize.sent_tokenize(decoded_sentences)
        generated_sentences.append(decoded_sentences_list[0]) # takes the first sentence 
        
    top_3_sentences = sort_by_similarity(full_sentence, generated_sentences)
    # top_sentences = sort_by_similarity(full_sentence, generated_sentences)
    # print(top_3_sentences)
    return top_3_sentences

def transe(str):
    zh_cn=translator.translate(str, src='en', dest='zh-cn').text
    eng=translator.translate(str, src='zh-cn', dest='en').text
    return str
#%% WH
# 정답 단어 추출

# https://www.nltk.org/book/ch05.html
def get_NP(passage):# 명사구에 해당하는 단어 추출(np:)
    answers = []
    trees = benepar_parser.parse_sents(sent_tokenize(passage))
    for sent_idx, tree in enumerate(trees):
        subtrees = tree.subtrees()
        for subtree in subtrees:
            if subtree.label() == "NP":
                answers.append(get_flattened(subtree))
    return list(set(answers))## 중복 제거

def remove_stopwords(stop_words, answers):
    aa=answers.copy()
    for ans in aa:
        if ans  in stop_words:
            aa.remove(ans)
    return aa

def ans_len_limit(answers):
    for i in answers:
        if len(i.split()) >=3:
            answers.remove(i)
    return answers
## 문제 평가
def encode_qa_pairs(questions, answers):
    SEQ_LENGTH = 512## 2000이었나? 아무튼 바꾸기(question.py의 question_generate 에도 있음)
    encoded_pairs = []
    for i in range(len(questions)):
        encoded_qa =qae_tokenizer(text=questions[i], text_pair=answers[i], padding="max_length", max_length=SEQ_LENGTH, truncation=True, return_tensors="pt")
        encoded_pairs.append(encoded_qa.to(DEVICE))
    return encoded_pairs
## 오답 생성
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


# https://www.nltk.org/book/ch05.html
def get_NN(distractor):#  NN, a noun
    NNs = []
    tree = benepar_parser.parse(distractor)
    subtrees = tree.subtrees()
    for subtree in subtrees:
        if subtree.label() in ["NN", "NNP", "NNS", "VB"]: #VB for edge case
            NNs.extend(subtree.leaves())       
    return NNs

# ## 정답이 포함된 문장 찾기
def get_sentence_with_ans(sentences, answer):
    target_sentence=[]
    for sentence in sentences:
        if answer in sentence:
            target_sentence.append(sentence)

