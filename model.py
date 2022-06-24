#%%
import warnings

import torch
from sentence_transformers import SentenceTransformer
from transformers import (AutoModelForSeq2SeqLM, AutoModelWithLMHead,
                          AutoTokenizer)

warnings.filterwarnings(action='ignore')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# %% load models
summarize_tokenizer = AutoTokenizer.from_pretrained("t5-small")
paraphrase_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

summarize_model = AutoModelWithLMHead.from_pretrained("t5-small")
paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
# add the EOS token as PAD token to avoid warnings
gpt2_model = AutoModelWithLMHead.from_pretrained("gpt2", pad_token_id=gpt2_tokenizer.eos_token_id)
bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

summarize_model.to(DEVICE)
paraphrase_model.to(DEVICE)
gpt2_model.to(DEVICE)
bert_model.to(DEVICE)
# %%
