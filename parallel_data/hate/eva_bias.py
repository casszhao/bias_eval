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
        model_name = 'dlb/electra-base-portuguese-uncased-brwac' #'neuralmind/bert-base-portuguese-cased'
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


def get_idt(lang):
    if lang == 'de':
        idt_list = ["Muslime", "Juden", "Juden", "Weiße", "Islam", "Schwarze", "Muslime", "Frauen", "Weiße", "Schwule", "Schwarze", "Democat", "Islamisch", "Allah", "Jüdisch", "Lesbe", "Transgender", "Rasse", "Braun", "Frau", "Mexikaner", "Religion", "Homosexuell", "Homosexualität", "Afrikaner"]
    elif lang == 'en':
        idt_list = ['muslim','jew', 'jews', 'white', 'islam', 'blacks', 'muslims', 'women', 'whites', 'gay', 'black', 'democat', 'islamic', 'allah', 'jewish','lesbian', 'transgender', 'race', 'brown', 'woman', 'mexican', 'religion', 'homosexual', 'homosexuality', 'africans']
    elif lang == 'ja': 
        idt_list = ['イスラム教徒','ユダヤ人', 'ユダヤ人', '白人', 'イスラム教', '黒人', 'イスラム教徒', '女性', '白人', 'ゲイ', '黒人', 'デモキャット', 'イスラム教', 'アッラー', 'ユダヤ人', 'レズビアン', 'トランスジェンダー', '人種', '茶色', '女性', 'メキシコ人', '宗教', '同性愛者', '同性愛', 'アフリカ人']
    elif lang == 'es': 
        idt_list = ['Musulmán', 'judío', 'judíos', 'blanco', 'islam', 'negros', 'musulmanes', 'mujeres', 'blancos', 'gay', 'negro', 'democat', 'islámico', 'Alá', 'judío', 'lesbiana', 'transgénero', 'raza', 'marrón', 'mujer', 'mexicano', 'religión', 'homosexual', 'homosexualidad', 'africanos']
    elif lang == 'pt':
        idt_list = ['muçulmanos', 'judeus', 'judeus', 'brancos', 'islâmicos', 'negros', 'muçulmanos', 'mulheres', 'brancos', 'gay', 'negro', 'democat', 'islâmico', 'allah', 'judeu', 'lésbicas', 'transgénero', 'raça', 'pardo', 'mulher', 'mexicano', 'religião', 'homossexual', 'homossexualidade', 'africanos']
    elif lang == 'fr': 
        idt_list = ['musulman', 'juif', 'juifs', 'blanc', 'islam', 'noirs', 'musulmans', 'femmes', 'blancs', 'gay', 'noir', 'démocat', 'islamique ', 'allah', 'juif', 'lesbienne', 'transgenre', 'race', 'marron', 'femme', 'mexicain', 'religion', 'homosexuel', 'homosexualité', 'africains']
    elif lang == 'du': 
        idt_list = ['moslim', 'jood', 'joden', 'blank', 'islam', 'zwarten', 'moslims', 'vrouwen', 'blanken', 'homo', 'zwart', 'demokat', 'islamitisch ', 'allah', 'joods', 'lesbisch', 'transgender', 'ras', 'bruin', 'vrouw', 'mexicaans', 'religie', 'homoseksueel', 'homoseksualiteit', 'afrikanen']
    elif lang == 'it': 
        idt_list = ['musulmano', 'ebreo', 'ebrei', 'bianco', 'islam', 'neri', 'musulmani', 'donne', 'bianchi', 'gay', 'neri', 'demogatti', 'islamici ', 'allah', 'ebrea', 'lesbica', 'transgender', 'razza', 'marrone', 'donna', 'messicana', 'religione', 'omosessuale', 'omosessualità', 'africani']
    elif lang == 'zh': 
        idt_list = ['穆斯林', '犹太', '犹太人', '白人', '伊斯兰', '黑人', '女', '白人', '工党', '真主', '犹太教', '拉拉', '变性', '种族', '棕色', '墨西哥', '宗教', '同性', '同性恋', '非洲人']
    elif lang == 'ar':
        idt_list = ["الله" ,"اليهود" ,"السحاقيات" ,"المتحولين جنسياً" ,"العرق" ,"البني" ,"المرأة" ,"المكسيكي" ,"الدين" ,"مثلي الجنس" ,"الشذوذ الجنسي" ,"الأفارقة"]
    return idt_list



model_name = get_model_name('en-mt5')
idt_list = get_idt('en')
file_name = 'en.json'


with open(file_name, 'r') as f:
    text_list = json.load(f)



# seperate idt and non idt
idt_token_list = []
token_list = []

for i in text_list:
    if len([each for each in idt_list if each.lower() in i.split()])>0: # hate speech with idt
        idt_token_list.append(i)
    else:
        token_list.append(i) # hate without idt / non hate with idt

each_corpus_number = min(len(idt_token_list),len(token_list))
print(len(idt_token_list),len(token_list),each_corpus_number)
idt_token_list = idt_token_list[:each_corpus_number]
token_list = token_list[:each_corpus_number]



tokenizer = AutoTokenizer.from_pretrained(model_name)
if 'mt5' in model_name:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                            output_hidden_states=True,
                                            output_attentions=True)
else:
    model = AutoModelForMaskedLM.from_pretrained(model_name,
                                            output_hidden_states=True,
                                            output_attentions=True)

model = model.eval()
if torch.cuda.is_available():
    model.to('cuda')

total_params = sum(param.numel() for param in model.parameters())
print(model_name)
print("==>> total_params: ", f"{total_params:,}")


total_score = 0
stereo_score = 0

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

mask_id = tokenizer.mask_token_id
log_softmax = torch.nn.LogSoftmax(dim=1)
attention = True #if args.method == 'aula' else False



# change to defined func required token tensor format
def get_scores_embed(tokens_list):
    hate_scores = []
    embes = []
    token_id_tensor = []
    for i in tokens_list:
        token_id_tensor.append(torch.tensor([tokenizer(i)['input_ids']]).to('cuda'))

    for tokens in token_id_tensor:
        with torch.no_grad():
            score, hidden_state = calculate_aul(model, tokens.to('cuda'), log_softmax, attention)
            hate_scores.append(score)
            embes.append(hidden_state)

    hate_scores = np.array(hate_scores)
    hate_scores = hate_scores.reshape([1, -1])
    embes = np.concatenate(embes)
    
    return hate_scores, embes

nonidt_hate_scores, nonidt_embes = get_scores_embed(token_list)
idt_hate_scores, idt_embes = get_scores_embed(idt_token_list)
# if bias  -----------> idt_hate_scores > nonidt_hate_scores
# if gender bias -----> male_scores > female_scores


bias_scores = nonidt_hate_scores < idt_hate_scores
weights = cos_sim(nonidt_hate_scores.T, idt_hate_scores)

weighted_bias_scores = bias_scores * weights
bias_score = np.sum(weighted_bias_scores) / np.sum(weights)
print('bias score (emb):', round(bias_score * 100, 2))