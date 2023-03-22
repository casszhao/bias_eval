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

def get_model_name_cased(lang_model):
    if lang_model == 'de-bert':
        model_name = "TurkuNLP/wikibert-base-de-cased" # 'deepset/gbert-base' 
    elif lang_model == 'de-distilbert':
        model_name ='distilbert-base-german-cased'



    elif lang_model == 'en-bert':
        model_name = 'bert-base-cased'
    elif lang_model == 'en-deberta':
        model_name = 'microsoft/deberta-v3-base'
    elif lang_model == 'en-distilbert':
        model_name = 'distilbert-base-cased'
    elif lang_model == 'en-roberta':
        model_name = 'roberta-base'
        
    elif lang_model == 'it-bert':
        model_name = "TurkuNLP/wikibert-base-it-cased" # "dbmdz/bert-base-italian-uncased"
    elif lang_model == 'it-xlm':
        model_name = "MilaNLProc/hate-ita-xlm-r-base"
    
    elif lang_model == 'es-bert': 
        model_name = "TurkuNLP/wikibert-base-es-cased" #'dccuchile/bert-base-spanish-wwm-uncased'

    elif lang_model == 'pt-bert': 
        model_name = "TurkuNLP/wikibert-base-pt-cased" # pablocosta/bertabaporu-base-uncased neuralmind/bert-base-portuguese-cased 
    elif lang_model == 'pt-xlm':
        model_name = "thegoodfellas/tgf-xlm-roberta-base-pt-br" 




    elif lang_model == 'ru-bert': 
        model_name = "TurkuNLP/wikibert-base-ru-cased" #'blinoff/roberta-base-russian-v0' 
    elif lang_model == 'ru-distilbert': 
        model_name = 'Geotrend/distilbert-base-ru-cased'

    elif lang_model == 'ja-bert': 
        model_name = "TurkuNLP/wikibert-base-ja-cased" #'cl-tohoku/bert-base-japanese-whole-word-masking'
    elif lang_model == 'ja-distilbert': 
        model_name = 'laboro-ai/distilbert-base-japanese'



    elif lang_model == 'ar-bert': # Arabic
        model_name = 'aubmindlab/bert-base-arabertv02' 

    elif lang_model == 'id-bert':
        model_name = 'cahya/bert-base-indonesian-1.5G'

    elif lang_model == 'zh-bert':
        model_name = 'hfl/chinese-bert-wwm-ext'




    elif lang_model == 'multi-bert':
        model_name = 'bert-base-multilingual-cased' # bert-base-multilingual-uncased
    elif lang_model == 'multi-deberta':
        model_name = 'microsoft/mdeberta-v3-base' # large 
    elif lang_model == 'multi-distilbert':
        model_name ='distilbert-base-multilingual-cased'
    return model_name

def get_model_name_uncased(lang_model):
    if lang_model == 'de-bert':
        model_name = 'deepset/gbert-base' 
    elif lang_model == 'de-distilbert':
        model_name ='distilbert-base-german-cased'



    elif lang_model == 'en-bert':
        model_name = 'bert-base-uncased'
    elif lang_model == 'en-deberta':
        model_name = 'microsoft/deberta-v3-base'
    elif lang_model == 'en-distilbert':
        model_name = 'distilbert-base-cased'
    elif lang_model == 'en-roberta':
        model_name = 'roberta-base'
        
    elif lang_model == 'it-bert':
        model_name = "dbmdz/bert-base-italian-uncased"
    elif lang_model == 'it-xlm':
        model_name = "MilaNLProc/hate-ita-xlm-r-base"
    
    elif lang_model == 'es-bert': 
        model_name = 'dccuchile/bert-base-spanish-wwm-uncased'

    elif lang_model == 'pt-bert': 
        model_name = "pablocosta/bertabaporu-base-uncased" # pablocosta/bertabaporu-base-uncased neuralmind/bert-base-portuguese-cased 
    elif lang_model == 'pt-xlm':
        model_name = "thegoodfellas/tgf-xlm-roberta-base-pt-br" 




    elif lang_model == 'ru-bert': 
        model_name = 'blinoff/roberta-base-russian-v0' 
    elif lang_model == 'ru-distilbert': 
        model_name = 'Geotrend/distilbert-base-ru-cased'

    elif lang_model == 'ja-bert': 
        model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    elif lang_model == 'ja-distilbert': 
        model_name = 'laboro-ai/distilbert-base-japanese'



    elif lang_model == 'ar-bert': # Arabic
        model_name = 'aubmindlab/bert-base-arabertv02' 

    elif lang_model == 'id-bert':
        model_name = 'cahya/bert-base-indonesian-1.5G'

    elif lang_model == 'zh-bert':
        model_name = 'hfl/chinese-bert-wwm-ext'




    elif lang_model == 'multi-bert':
        model_name = 'bert-base-multilingual-uncased' # bert-base-multilingual-uncased
    elif lang_model == 'multi-deberta':
        model_name = 'microsoft/mdeberta-v3-base' # large 
    elif lang_model == 'multi-distilbert':
        model_name ='distilbert-base-multilingual-cased'
    return model_name


lang = 'ar'
model = 'bert' # mdeberta
mono = True # True False
second_set = False
cased_model = False

if mono == True:
    if cased_model == True:
        model_name = get_model_name_cased(lang + '-' + model)
    else:
        model_name = get_model_name_uncased(lang + '-' + model) # "multi-bert" lang
else:
    if  cased_model == True:
        model_name = get_model_name_cased('multi-' + model) # "multi-bert"
    else:
        model_name = get_model_name_uncased('multi-' + model) # "multi-bert"




# df = pd.read_json('../translated_data/russian.json')
# disadv_text_list = list(df['anti-stereotype'])
# adv_text_list = list(df['stereotype'])

disadv = 'hate_nonidt' # nonhate_idt hate_nonidt

if second_set == True:
    adv_corpus = f'./hateB/{lang}/hate_idt.json'
    disadv_corpus = f'./hateB/{lang}/{disadv}.json'
else: 
    adv_corpus = f'./hate/{lang}/hate_idt.json'
    disadv_corpus = f'./hate/{lang}/{disadv}.json'

with open(adv_corpus, 'r') as f:
    adv_text_list = json.load(f)
with open(disadv_corpus, 'r') as f:
    disadv_text_list = json.load(f)







each_corpus_number = min(len(adv_text_list),len(disadv_text_list))
print(' ')
print('adv data number / disadv data number / each corpus numbers', len(adv_text_list),len(disadv_text_list), each_corpus_number)
print(' ')
adv_text_list = adv_text_list[:each_corpus_number]
disadv_text_list = disadv_text_list[:each_corpus_number]


if "TurkuNLP" in model_name:
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

else: 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name,
                                            output_hidden_states=True,
                                            output_attentions=True)

model = model.eval()
if torch.cuda.is_available():
    model.to('cuda')

total_params = sum(param.numel() for param in model.parameters())


print(' ')
if mono == True:
    print(' Mono lingual !   ', model_name)
else: 
    print(' Multi lingual !   ', model_name)

print("==>> total_params: ", f"{total_params:,}")


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

mask_id = tokenizer.mask_token_id
log_softmax = torch.nn.LogSoftmax(dim=1)
attention = True #if args.method == 'aula' else False





adv_scores, adv_embes = get_scores_embed(adv_text_list)
disadv_scores, disadv_embes = get_scores_embed(disadv_text_list)
# if bias  -----------> idt_hate_scores > nonidt_hate_scores
# if gender bias -----> male_scores > female_scores


bias_scores = adv_scores > disadv_scores
weights = cos_sim(disadv_embes, adv_embes.T)


weighted_bias_scores = bias_scores * weights
bias_score = np.sum(weighted_bias_scores) / np.sum(weights)


print('each corpus numbers', each_corpus_number)
print('model_name : ', model_name)
print('language and corpus -->', lang)
print('bias score (emb):', round(bias_score * 100, 2))