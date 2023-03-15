import json
import pandas as pd
import argparse
import torch
import difflib
import nltk
import regex as re
import numpy as np
import MeCab
import pickle

from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForSeq2SeqLM


def calculate_aul(model, token_ids, log_softmax, attention):
    '''
    Given token ids of a sequence, return the averaged log probability of
    unmasked sequence (AULA or AUL).
    '''
    output = model(token_ids)
    logits = output.logits.squeeze(0)
    log_probs = log_softmax(logits)
    token_ids = token_ids.view(-1, 1).detach()
    token_log_probs = log_probs.gather(1, token_ids)[1:-1]
    if attention:
        attentions = torch.mean(torch.cat(output.attentions, 0), 0)
        averaged_attentions = torch.mean(attentions, 0)
        averaged_token_attentions = torch.mean(averaged_attentions, 0)
        token_log_probs = token_log_probs.squeeze(1) * averaged_token_attentions[1:-1]
    sentence_log_prob = torch.mean(token_log_probs)
    score = sentence_log_prob.item()

    hidden_states = output.hidden_states[-1][:,1:-1]
    hidden_state = torch.mean(hidden_states, 1).detach().cpu().numpy()

    return score, hidden_state

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def get_model_name(lang):
    if lang == 'de':
        model_name = 'deepset/gbert-base' 
    elif lang == 'de-xlm':
        model_name = 'xlm-roberta-large-finetuned-conll03-german'
    elif lang == 'en':
        model_name = 'bert-base-uncased'
    elif lang == 'en-mt5':
        model_name = 'google/mt5-base'
    elif lang == 'en-roberta':
        model_name = 'xlm-roberta-base'
    elif lang == 'en-xlm':
        model_name = 'xlm-mlm-ende-1024'
    elif lang == 'ja': 
        model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    elif lang == 'ar': # Arabic
        model_name = 'aubmindlab/bert-base-arabertv02' 
    elif lang == 'ar-xlm': # Arabic
        model_name = '3ebdola/Dialectal-Arabic-XLM-R-Base'
    elif lang == 'es': 
        model_name = 'dccuchile/bert-base-spanish-wwm-uncased'
    elif lang == 'es-xlm': 
        model_name = 'MMG/xlm-roberta-large-ner-spanish' 
    elif lang == 'pt': 
        model_name = 'pablocosta/bertabaporu-base-uncased' #'neuralmind/bert-base-portuguese-cased'
    elif lang == 'pt-xlm':
        model_name = "thegoodfellas/tgf-xlm-roberta-base-pt-br" 
    elif lang == 'ru': 
        model_name = 'blinoff/roberta-base-russian-v0'
    elif lang == 'id':
        model_name = 'cahya/bert-base-indonesian-1.5G'
    elif lang == 'zh':
        model_name = 'hfl/chinese-bert-wwm-ext'
    elif lang == 'it':
        model_name = "dbmdz/bert-base-italian-uncased"
    elif lang == 'it-xlm':
        model_name = "MilaNLProc/hate-ita-xlm-r-base"
    elif lang == 'multi-xlm':
        model_name = 'xlm-mlm-100-1280'
    elif lang == 'multi-bert':
        model_name = 'bert-base-multilingual-uncased'
    elif lang == 'multi-mt5':
        model_name = 'csebuetnlp/mT5_multilingual_XLSum'
    return model_name


lang = 'de'
model_name = get_model_name(lang) # "multi-bert" lang


# df = pd.read_json('../translated_data/russian.json')
# disadv_text_list = list(df['anti-stereotype'])
# adv_text_list = list(df['stereotype'])

adv_corpus = f'./gender/{lang}/male.json'
disadv_corpus = f'./gender/{lang}/female.json'

########### adv > disadv as expected
with open(adv_corpus, 'r') as f:
    adv_text_list = json.load(f)
with open(disadv_corpus, 'r') as f:
    disadv_text_list = json.load(f)



each_corpus_number = min(len(adv_text_list),len(disadv_text_list))
print(len(adv_text_list),len(disadv_text_list),each_corpus_number)
adv_text_list = adv_text_list[:each_corpus_number]
disadv_text_list = disadv_text_list[:each_corpus_number]



tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name,
                                            output_hidden_states=True,
                                            output_attentions=True)

model = model.eval()
if torch.cuda.is_available():
    model.to('cuda')

total_params = sum(param.numel() for param in model.parameters())
print(model_name)
print("==>> total_params: ", f"{total_params:,}")



if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

mask_id = tokenizer.mask_token_id
log_softmax = torch.nn.LogSoftmax(dim=1)
attention = True #if args.method == 'aula' else False



# change to defined func required token tensor format
def get_scores_embed(tokens_list):
    scores = []
    embes = []
    token_id_tensor = []
    for i in tokens_list:
        token_id_tensor.append(torch.tensor([tokenizer(i)['input_ids']]).to('cuda')) # get ids

    for tokens in token_id_tensor:
        with torch.no_grad():
            score, hidden_state = calculate_aul(model, tokens.to('cuda'), log_softmax, attention)
            scores.append(score)
            embes.append(hidden_state)

    scores = np.array(scores)
    scores = scores.reshape([1, -1])
    embes = np.concatenate(embes)
    
    return scores, embes

adv_scores, adv_embes = get_scores_embed(adv_text_list)
disadv_scores, disadv_embes = get_scores_embed(disadv_text_list)
# if bias  -----------> idt_hate_scores > nonidt_hate_scores
# if gender bias -----> male_scores > female_scores


bias_scores = adv_scores > disadv_scores
weights = cos_sim(disadv_embes, adv_embes.T)


weighted_bias_scores = bias_scores * weights
bias_score = np.sum(weighted_bias_scores) / np.sum(weights)
print('bias score (emb):', round(bias_score * 100, 2))