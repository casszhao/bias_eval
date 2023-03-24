import json
import pandas as pd
import argparse
import torch
import random
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
def get_scores_embed(tokens_list, tokenizer, model):
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


attention = True
from func import get_model_name_uncased, get_model_name_cased

parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str, #required=True, choices=['en', 'de', 'ja', 'ar', 'es', 'pt', 'ru', 'id', 'zh'],
                    help='Path to evaluation dataset.',
                    default='pt')
parser.add_argument('--method', type=str, #required=True, choices=['aula', 'aul'],
                    default='aula')
parser.add_argument('--if_cased', type=str, #required=True, choices=['cased', 'uncased'],
                    default='uncased')
parser.add_argument('--log_name', type=str)

args = parser.parse_args()

model = 'bert' # mdeberta


print(' ')
print(' ')
print(' ')
print(' ')
print(' -------- ', str(args.lang))

import os
pwd = os.getcwd()
print(pwd)
adv_corpus = str(pwd) + f'/parallel_data/hate/{args.lang}/adv_input_list.json'
disadv_corpus = str(pwd) + f'/parallel_data/hate/{args.lang}/disadv_input_list.json' 

with open(adv_corpus, 'r') as f:
    adv_text_list = json.load(f)
with open(disadv_corpus, 'r') as f:
    disadv_text_list = json.load(f)





############# for mono first

if args.if_cased == 'cased': model_name_mono = get_model_name_cased(args.lang + '-' + model)
else: model_name_mono = get_model_name_uncased(args.lang + '-' + model) # "multi-bert" lang
print(model_name_mono)

if "TurkuNLP" in model_name_mono:
    from transformers import BertTokenizer
    tokenizer_mono = BertTokenizer.from_pretrained(model_name_mono)
else: tokenizer_mono = AutoTokenizer.from_pretrained(model_name_mono)
model_mono = AutoModelForMaskedLM.from_pretrained(model_name_mono,
                                            output_hidden_states=True,
                                            output_attentions=True)

model_mono = model_mono.eval()
if torch.cuda.is_available():
    model_mono.to('cuda')

total_params_mono = sum(param.numel() for param in model_mono.parameters())
print("==>> total_params: ", f"{total_params_mono:,}")

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

mask_id_mono = tokenizer_mono.mask_token_id
log_softmax = torch.nn.LogSoftmax(dim=1)


adv_scores, adv_embes, adv_token_len = get_scores_embed(adv_text_list, tokenizer_mono, model_mono)
disadv_scores, disadv_embes, disadv_token_len = get_scores_embed(disadv_text_list, tokenizer_mono, model_mono)
avg_token_num_mono = int((adv_token_len+disadv_token_len)/2)

bias_scores = adv_scores > disadv_scores
weights = cos_sim(disadv_embes, adv_embes.T)
weighted_bias_scores = bias_scores * weights
bias_score = np.sum(weighted_bias_scores) / np.sum(weights)
bias_score_mono = round(bias_score * 100, 2)
print("==>> (bias_score_mono): ", bias_score_mono)

from statsmodels.stats.api import SquareTable
def sig_test(bias_scores, weighted_bias_scores, weights):
    shape = bias_scores.shape
    bias_scores_random = [random.choice([True, False]) for _ in range(shape[1])]
    bias_scores_random = np.array(bias_scores_random)
    bias_scores_random = bias_scores_random.reshape([1, -1])
    weighted_bias_scores_random = bias_scores_random * weights

    bias_list = list(map(bool,weighted_bias_scores[0]*100))
    bias_list_random = list(map(bool,weighted_bias_scores_random[0]*100))
    df = pd.DataFrame(list(zip(bias_list_random, bias_list)), columns =['Random', 'MBE'])
    myCross = pd.crosstab(df['Random'], df['MBE'])
    stats = SquareTable(myCross, shift_zeros=False).symmetry()
    print(stats)
    chi_squared = stats.statistic
    p_value = stats.pvalue
    degree_freedom = stats.df
    return p_value, chi_squared, degree_freedom


p_value_mono, chi_squared_mono, degree_freedom_mono = sig_test(bias_scores, weighted_bias_scores, weights)











############# for multi 
if args.if_cased == 'cased':
    model_name = 'bert-base-multilingual-cased'
else:
    model_name = 'bert-base-multilingual-uncased'
print(model_name)

if "TurkuNLP" in model_name:
    tokenizer = BertTokenizer.from_pretrained(model_name)
else: 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name,
                                            output_hidden_states=True,
                                            output_attentions=True)



model = model.eval()
if torch.cuda.is_available():
    model.to('cuda')

total_params_multi = sum(param.numel() for param in model.parameters())
print("==>> total_params: ", f"{total_params_multi:,}")


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

mask_id = tokenizer.mask_token_id
log_softmax = torch.nn.LogSoftmax(dim=1)


adv_scores, adv_embes, adv_token_len = get_scores_embed(adv_text_list, tokenizer, model)
disadv_scores, disadv_embes, disadv_token_len = get_scores_embed(disadv_text_list, tokenizer, model)
avg_token_num_multi = int((adv_token_len+disadv_token_len)/2)

bias_scores = adv_scores > disadv_scores
weights = cos_sim(disadv_embes, adv_embes.T)
weighted_bias_scores = bias_scores * weights
bias_score = np.sum(weighted_bias_scores) / np.sum(weights)
bias_score_multi = round(bias_score * 100, 2)
print("==>> (bias_score_multi): ", bias_score_multi)

p_value, chi_squared, degree_freedom = sig_test(bias_scores, weighted_bias_scores, weights)
#  "Language,Corpus Size,Monolingual,Multilingual,Diff_in_Scores,
#  MonoModel_Size,MultiModel_Size,Mono_token_len, Multi_token_len"

with open(str(args.log_name), 'a') as writer:
    writer.write(str(args.lang))
    writer.write(',')
    writer.write(str(len(adv_text_list)))
    writer.write(',')
    writer.write(str(bias_score_mono))
    writer.write(',')
    writer.write(str(bias_score_multi))
    writer.write(',')
    writer.write(str("{:.2f}".format(round(bias_score_mono-bias_score_multi, 2))))
    writer.write(',')
    writer.write(str(total_params_mono))
    writer.write(',')
    writer.write(str(total_params_multi))
    writer.write(',')
    writer.write(str(avg_token_num_mono))
    writer.write(',')
    writer.write(str(avg_token_num_multi))
    writer.write(',')
    writer.write(str(p_value_mono))
    writer.write(',')
    writer.write(str(p_value))
    writer.write('\n')