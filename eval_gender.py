import json
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
from transformers import AutoModelForMaskedLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, #required=True,
                        choices=['en', 'de', 'ja', 'ar', 'es', 'pt', 'ru', 'id', 'zh'],
                        help='Path to evaluation dataset.',
                        default='zh')
    parser.add_argument('--method', type=str, #required=True,
                        choices=['aula', 'aul'],
                        default='aula')
    parser.add_argument('--corpus', type=str, #required=True,
                        choices=['ted', 'news'],
                        default='ted')
    parser.add_argument('--if_cased', type=str, #required=True,
                        choices=['cased', 'uncased'],
                        default='uncased')
    parser.add_argument('--if_multilingual', type=str, #required=True,
                        choices=['multi', 'mono'],
                        default='multi')
    args = parser.parse_args()

    return args


def load_tokenizer_and_model(args):
    '''
    Load tokenizer and model to evaluate.
    '''
    if args.lang == 'de':
        model_name = 'deepset/gbert-base'
    elif args.lang == 'ja':
        model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    elif args.lang == 'ar':
        model_name = 'aubmindlab/bert-base-arabertv02'
    elif args.lang == 'es':
        model_name = 'dccuchile/bert-base-spanish-wwm-uncased'
    elif args.lang == 'pt':
        model_name = 'neuralmind/bert-base-portuguese-cased'
    elif args.lang == 'ru':
        model_name = 'blinoff/roberta-base-russian-v0'
    elif args.lang == 'id':
        model_name = 'cahya/bert-base-indonesian-1.5G'
    elif args.lang == 'zh':
        model_name = 'hfl/chinese-bert-wwm-ext'
    elif args.lang == 'multi-xlm':
        model_name = 'xlm-mlm-100-1280'
    elif args.lang == 'multi-bert':
        model_name = 'bert-base-multilingual-uncased'

    model = AutoModelForMaskedLM.from_pretrained(model_name,
                                                 output_hidden_states=True,
                                                 output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    return tokenizer, model


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


def convert_ids(female_inputs, tokenizer, tokenizer_2):
    temp_female_inputs = []
    for ids in female_inputs:
        text = tokenizer.convert_ids_to_tokens(ids[0])
        text = text[1:-1]
        if args.lang == 'zh': text = ''.join(text)
        else: text = ' '.join(text)
        text = torch.tensor([tokenizer_2(text)['input_ids']]).to('cuda')
        temp_female_inputs.append(text)
    return temp_female_inputs

def main(args):
    '''
    Evaluate the bias in masked language models.
    '''
    tokenizer, model = load_tokenizer_and_model(args)
    total_score = 0
    stereo_score = 0
    corpus = args.corpus
    lang = args.lang

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    mask_id = tokenizer.mask_token_id
    log_softmax = torch.nn.LogSoftmax(dim=1)

    female_inputs = pickle.load(open(f'parallel_data/{corpus}/{lang}_f.bin', 'rb'))
    male_inputs = pickle.load(open(f'parallel_data/{corpus}/{lang}_m.bin', 'rb'))

    # decode back to text


    from func import get_model_name_uncased, get_model_name_cased

    if args.if_multilingual == 'mono':
        if args.if_cased == 'cased': model_name = get_model_name_cased(lang + '-bert')
        else: model_name = get_model_name_uncased(lang + '-bert') # "multi-bert" lang
    elif args.if_multilingual == 'multi':
        if args.if_cased == 'cased': model_name = get_model_name_cased('multi-bert') # "multi-bert"
        else: model_name = get_model_name_uncased('multi-bert') # "multi-bert"

    
    if "TurkuNLP" in model_name:
        from transformers import BertTokenizer
        tokenizer_2 = BertTokenizer.from_pretrained(model_name)

    else: 
        tokenizer_2 = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name,
                                                output_hidden_states=True,
                                                output_attentions=True)

    model = model.eval()
    if torch.cuda.is_available():
        model.to('cuda')
    
    female_inputs = convert_ids(female_inputs, tokenizer, tokenizer_2)
    male_inputs = convert_ids(male_inputs, tokenizer, tokenizer_2)


    attention = True if args.method == 'aula' else False

    female_scores = []
    male_scores = []
    female_embes = []
    male_embes = []

    for female_tokens in female_inputs:
        with torch.no_grad():
            female_score, female_hidden_state = calculate_aul(model, female_tokens, log_softmax, attention)
            female_scores.append(female_score)
            female_embes.append(female_hidden_state)

    for male_tokens in male_inputs:
        with torch.no_grad():
            male_score, male_hidden_state = calculate_aul(model, male_tokens, log_softmax, attention)
            male_scores.append(male_score)
            male_embes.append(male_hidden_state)

    female_scores = np.array(female_scores)
    female_scores = female_scores.reshape([-1, 1])
    male_scores = np.array(male_scores)
    male_scores = male_scores.reshape([1, -1])
    bias_scores = male_scores > female_scores 

    female_embes = np.concatenate(female_embes)
    male_embes = np.concatenate(male_embes)
    weights = cos_sim(female_embes, male_embes.T)

    weighted_bias_scores = bias_scores * weights
    bias_score = np.sum(weighted_bias_scores) / np.sum(weights)
    print('bias score (emb):', round(bias_score * 100, 2))


if __name__ == "__main__":
    args = parse_args()
    main(args)
