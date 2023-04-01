import json
import pandas as pd
import argparse
import torch
import random
import os
import difflib
import nltk
import regex as re
import numpy as np
import MeCab
import pickle
from random import sample
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModelForSeq2SeqLM, BertTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str, #required=True, choices=['en', 'de', 'ja', 'ar', 'es', 'pt', 'ru', 'id', 'zh'],
                    help='Path to evaluation dataset.',
                    default='es')
parser.add_argument('--method', type=str, #required=True, choices=['aula', 'aul'],
                    default='aula')
parser.add_argument('--if_cased', type=str, #required=True, choices=['cased', 'uncased'],
                    default='cased')
parser.add_argument('--dataset_name', type=str, default='hate')
parser.add_argument('--model', type=str, default='bert')
parser.add_argument('--save_results', type=str, default='subgroup_results.csv')


args = parser.parse_args()

model = args.model # mdeberta


def get_tops(arr_2d, top_k):
    arr_1d = arr_2d.flatten()
    sorted_arr = np.sort(arr_1d)[::-1]
    top_3_max = sorted_arr[:top_k]
    top_3_max_indices = np.argsort(arr_1d)[::-1][:top_k]
    # convert the flat indices to the indices of the original array shape
    top_3_max_indices_original = np.unravel_index(top_3_max_indices, arr_2d.shape)
    return top_3_max_indices_original

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
    return scores, embes, int(avg_token_num/len(tokens_list))

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
    chi_squared = stats.statistic
    p_value = stats.pvalue
    degree_freedom = stats.df
    return p_value, chi_squared, degree_freedom

attention = True
from func import get_model_name_uncased, get_model_name_cased






# pwd = os.getcwd()

idt_dic_name = f'parallel_data/hate/{args.lang}/idt_dict.pkl'
with open(idt_dic_name, 'rb') as f:
    idt_dict = pickle.load(f)
print(idt_dict)

hate_templ_name = f'parallel_data/hate/{args.lang}/hate_templ.json'
with open(hate_templ_name, 'r') as f:
    hate_templ_list = json.load(f)

nonhate_templ_name = f'parallel_data/hate/{args.lang}/nonhate_templ.json'
with open(nonhate_templ_name, 'r') as f:
    nonhate_templ_list = json.load(f)

templ_num = min(len(hate_templ_list), len(nonhate_templ_list))
hate_templ_list = hate_templ_list[:templ_num]
nonhate_templ_list = nonhate_templ_list[:templ_num]
assert len(hate_templ_list) == len(nonhate_templ_list)



from random import sample
import random


if args.lang == 'en': neural_idt = 'people'
elif args.lang == 'de': neural_idt = 'Leute'
elif args.lang == 'du': neural_idt = 'mensen'
elif args.lang == 'es': neural_idt = 'gente'
elif args.lang == 'fr': neural_idt = 'pessoas'
elif args.lang == 'hi': neural_idt = 'log' # janata
elif args.lang == 'it': neural_idt = 'persone'
elif args.lang == 'pt': neural_idt = 'pessoas'
elif args.lang == 'zh': neural_idt = '人'
elif args.lang == 'ar': neural_idt = 'الناس'
elif args.lang == 'po': neural_idt = 'ludzie'
else: 
    print(' no neural words!!!!')


def create_corpus(target, idt_dict, hate_templ_list, nonhate_templ_list, neural_idt):
    hate_idt = []
    nonhate_idt = []
    hate_nonidt = []
    nonhate_nonidt = []

    if target == 'all':
            
        idt_temp_list = []
        
        for case_templ in hate_templ_list:
            idt = random.sample(list(idt_dict.values()), 1)[0]
            idt_temp_list.append(idt)

            input = re.sub("\[[^\]]*\]", idt, case_templ)
            hate_idt.append(input)

            input = re.sub("\[[^\]]*\]", neural_idt, case_templ)
            hate_nonidt.append(input)

        for i, non_case_templ in enumerate(nonhate_templ_list):
            input = re.sub("\[[^\]]*\]", idt_temp_list[i], non_case_templ)
            nonhate_idt.append(input)

            input = re.sub("\[[^\]]*\]", neural_idt, non_case_templ)
            nonhate_nonidt.append(input)
        
        #print(' temp idt list: ', idt_temp_list)

    else:
        for case_templ in hate_templ_list:
            input = re.sub("\[[^\]]*\]", target, case_templ)
            hate_idt.append(input)
            input = re.sub("\[[^\]]*\]", neural_idt, case_templ)
            hate_nonidt.append(input)
        for non_case_templ in nonhate_templ_list:
            input = re.sub("\[[^\]]*\]", target, non_case_templ)
            nonhate_idt.append(input)
            input = re.sub("\[[^\]]*\]", neural_idt, non_case_templ)
            nonhate_nonidt.append(input)

    return hate_idt,nonhate_idt,hate_nonidt,nonhate_nonidt


def get_scores(adv_text_list, disadv_text_list, tokenizer, model):

    adv_scores, adv_embes, adv_token_len = get_scores_embed(adv_text_list, tokenizer, model)
    disadv_scores, disadv_embes, disadv_token_len = get_scores_embed(disadv_text_list, tokenizer, model)
    avg_token_num = int((adv_token_len+disadv_token_len)/2)

    bias_scores = adv_scores > disadv_scores
    weights = cos_sim(disadv_embes, adv_embes.T)
    weighted_bias_scores = bias_scores * weights
    bias_score = np.sum(weighted_bias_scores) / np.sum(weights)
    bias_score_final = round(bias_score * 100, 2)

    p_value, chi_squared, degree_freedom = sig_test(bias_scores, weighted_bias_scores, weights)

    top_index_1d, top_index_2d = get_tops(weighted_bias_scores, 3)

    return bias_score_final, p_value, avg_token_num, adv_text_list[top_index_1d[0]],disadv_text_list[top_index_2d[0]]


def write(disadv_corpus_name):
    with open(args.save_results, 'a+') as writer:
        writer.write(str(args.lang))
        writer.write(';')
        writer.write(str((mono_multi)))
        writer.write(';')
        if bias_target != 'all':writer.write(str(one_target))
        else:writer.write('all')
        writer.write(';')
        writer.write(disadv_corpus_name)
        writer.write(';')
        writer.write(str(bias_score))
        writer.write(';')
        writer.write(str(p_value))
        writer.write(';')
        writer.write(str(most_biasedpair_hate_idt))
        writer.write(';')
        writer.write(str(most_biasedpair_disadv))
        writer.write(';')
        writer.write(str(token_num))
        writer.write(';')
        writer.write(str(len(hate_idt)))
        writer.write('\n')



# do mono
mono_multi = 'mono'

if args.if_cased == 'cased': model_name = get_model_name_cased(args.lang + '-' + model)
else: model_name = get_model_name_uncased(args.lang + '-' + model) # "multi-bert" lang


if "TurkuNLP" in model_name:
    tokenizer = BertTokenizer.from_pretrained(model_name)
else: 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name,output_hidden_states=True,output_attentions=True)
model = model.eval()
if torch.cuda.is_available(): model.to('cuda'), torch.set_default_tensor_type('torch.cuda.FloatTensor')
mask_id = tokenizer.mask_token_id
log_softmax = torch.nn.LogSoftmax(dim=1)



bias_target = 'all'
hate_idt,nonhate_idt,hate_nonidt,nonhate_nonidt = create_corpus(str(bias_target), idt_dict, hate_templ_list, nonhate_templ_list,neural_idt)

bias_score, p_value, token_num, most_biasedpair_hate_idt,most_biasedpair_disadv = get_scores(hate_idt, nonhate_idt, tokenizer, model)
write('nonhate_idt')
bias_score, p_value, token_num, most_biasedpair_hate_idt,most_biasedpair_disadv = get_scores(hate_idt, hate_nonidt, tokenizer, model)
write('hate_nonidt')
bias_score, p_value, token_num, most_biasedpair_hate_idt,most_biasedpair_disadv = get_scores(hate_idt, nonhate_nonidt, tokenizer, model)
write('nonhate_nonidt')


for one_target in idt_dict:
    bias_target = idt_dict.get(one_target)
    
    hate_idt,nonhate_idt,hate_nonidt,nonhate_nonidt = create_corpus(str(bias_target), idt_dict, hate_templ_list, nonhate_templ_list,neural_idt)
    bias_score, p_value, token_num, most_biasedpair_hate_idt,most_biasedpair_disadv = get_scores(hate_idt, nonhate_idt, tokenizer, model)
    write('nonhate_idt')
    bias_score, p_value, token_num, most_biasedpair_hate_idt,most_biasedpair_disadv = get_scores(hate_idt, hate_nonidt, tokenizer, model)
    write('hate_nonidt')
    bias_score, p_value, token_num, most_biasedpair_hate_idt,most_biasedpair_disadv = get_scores(hate_idt, nonhate_nonidt, tokenizer, model)
    write('nonhate_nonidt')



# do multi
mono_multi = 'multi'

if args.if_cased == 'cased': model_name = 'bert-base-multilingual-cased'
else: model_name = 'bert-base-multilingual-uncased'

if "TurkuNLP" in model_name:
    tokenizer = BertTokenizer.from_pretrained(model_name)
else: 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name,output_hidden_states=True,output_attentions=True)
model = model.eval()
if torch.cuda.is_available(): model.to('cuda'), torch.set_default_tensor_type('torch.cuda.FloatTensor')
mask_id = tokenizer.mask_token_id
log_softmax = torch.nn.LogSoftmax(dim=1)



bias_target = 'all'
hate_idt,nonhate_idt,hate_nonidt,nonhate_nonidt = create_corpus(str(bias_target), idt_dict, hate_templ_list, nonhate_templ_list,neural_idt)

bias_score, p_value, token_num, most_biasedpair_hate_idt,most_biasedpair_disadv = get_scores(hate_idt, nonhate_idt, tokenizer, model)
write('nonhate_idt')
bias_score, p_value, token_num, most_biasedpair_hate_idt,most_biasedpair_disadv = get_scores(hate_idt, hate_nonidt, tokenizer, model)
write('hate_nonidt')
bias_score, p_value, token_num, most_biasedpair_hate_idt,most_biasedpair_disadv = get_scores(hate_idt, nonhate_nonidt, tokenizer, model)
write('nonhate_nonidt')


for one_target in idt_dict:
    bias_target = idt_dict[one_target]
    hate_idt,nonhate_idt,hate_nonidt,nonhate_nonidt = create_corpus(str(bias_target), idt_dict, hate_templ_list, nonhate_templ_list,neural_idt)
    bias_score, p_value, token_num, most_biasedpair_hate_idt,most_biasedpair_disadv = get_scores(hate_idt, nonhate_idt, tokenizer, model)
    write('nonhate_idt')
    bias_score, p_value, token_num, most_biasedpair_hate_idt,most_biasedpair_disadv = get_scores(hate_idt, hate_nonidt, tokenizer, model)
    write('hate_nonidt')
    bias_score, p_value, token_num, most_biasedpair_hate_idt,most_biasedpair_disadv = get_scores(hate_idt, nonhate_nonidt, tokenizer, model)
    write('nonhate_nonidt')


    
# total_params_multi = sum(param.numel() for param in model.parameters())
