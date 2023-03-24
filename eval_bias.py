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
from random import sample
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
    avg_token_num = 0
    for i in tokens_list:
        input_ids = tokenizer(i)['input_ids']
        avg_token_num += len(input_ids)
        token_id_tensor.append(torch.tensor([input_ids]).to('cuda')) # get ids

    for tokens in token_id_tensor:
        with torch.no_grad():
            score, hidden_state = calculate_aul(model, tokens.to('cuda'), log_softmax, attention)
            scores.append(score)
            embes.append(hidden_state)

    scores = np.array(scores)
    scores = scores.reshape([1, -1])
    embes = np.concatenate(embes)
    print(' avg_token_num: ', int(avg_token_num/len(tokens_list)))
    return scores, embes, int(avg_token_num/len(tokens_list))



from func import get_model_name_uncased, get_model_name_cased

parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str, #required=True,
                    # choices=['en', 'de', 'ja', 'ar', 'es', 'pt', 'ru', 'id', 'zh'],
                    help='Path to evaluation dataset.',
                    default='de')
parser.add_argument('--method', type=str, #required=True,
                    # choices=['aula', 'aul'],
                    default='aula')
parser.add_argument('--if_cased', type=str, #required=True,
                    # choices=['cased', 'uncased'],
                    default='uncased')
parser.add_argument('--if_multilingual', type=str, #required=True,
                    choices=['multi', 'mono'],
                    default='multi')

args = parser.parse_args()

model = 'bert' # mdeberta


if args.if_multilingual == 'mono':
    if args.if_cased == 'cased':
        model_name = get_model_name_cased(args.lang + '-' + model)
    else:
        model_name = get_model_name_uncased(args.lang + '-' + model) # "multi-bert" lang
else:
    if args.if_cased == 'cased':
        model_name = get_model_name_cased('multi-' + model) # "multi-bert"
    else:
        model_name = get_model_name_uncased('multi-' + model) # "multi-bert"


adv = 'hate_idt'
disadv = 'nonhate_idt' # nonhate_idt hate_nonidt

import os
pwd = os.getcwd()
print(pwd)
adv_corpus = str(pwd) + f'/parallel_data/hate/{args.lang}/{adv}.json'
disadv_corpus = str(pwd) + f'/parallel_data/hate/{args.lang}/{disadv}.json' 

with open(adv_corpus, 'r') as f:
    adv_text_list = json.load(f)
with open(disadv_corpus, 'r') as f:
    disadv_text_list = json.load(f)



each_corpus_number = min(len(adv_text_list),len(disadv_text_list))
print(' ')
print('adv data number / disadv data number / each corpus numbers', len(adv_text_list),len(disadv_text_list), each_corpus_number)
print(' ')
adv_text_list = sample(adv_text_list,each_corpus_number)
disadv_text_list = sample(disadv_text_list,each_corpus_number)


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


print("==>> total_params: ", f"{total_params:,}")


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

mask_id = tokenizer.mask_token_id
log_softmax = torch.nn.LogSoftmax(dim=1)
attention = True #if args.method == 'aula' else False





adv_scores, adv_embes, adv_token_len = get_scores_embed(adv_text_list)
disadv_scores, disadv_embes, disadv_token_len = get_scores_embed(disadv_text_list)
print(' total avg_token_num: ', int((adv_token_len+disadv_token_len)/2))
# if bias  -----------> idt_hate_scores > nonidt_hate_scores
# if gender bias -----> male_scores > female_scores

# bias_scores = male_scores > female_scores 
bias_scores = adv_scores > disadv_scores
# weights = cos_sim(female_embes, male_embes.T)
weights = cos_sim(disadv_embes, adv_embes.T)


weighted_bias_scores = bias_scores * weights
bias_score = np.sum(weighted_bias_scores) / np.sum(weights)
bias_score = round(bias_score * 100, 2)


print('each corpus numbers', each_corpus_number)
print('model_name : ', model_name)
print('language and corpus -->', args.lang)
print('bias score (emb):', bias_score)


with open('./results/idt.txt', 'a') as writer:
    writer.write('\n' )
    writer.write('\n' )
    writer.write(str(args))
    writer.write(' model size:' + str(f"{total_params:,}"))
    writer.write('\n' )
    writer.write('num ' + str(each_corpus_number))
    writer.write('\n' )
    writer.write('id_length ' )
    writer.write(str(int((adv_token_len+disadv_token_len)/2)) + ' ')
    writer.write('\n')
    writer.write(model_name + ' ')
    writer.write(f'  bias score (emb): {bias_score}')