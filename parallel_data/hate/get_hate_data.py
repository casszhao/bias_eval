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
import re
import os


from random import sample
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset_builder, load_dataset, get_dataset_split_names, get_dataset_config_names
from deep_translator import GoogleTranslator, MicrosoftTranslator, DeeplTranslator

lang = 'hi'

def get_dataset_name(lang):
    if lang == 'zh':
        dataset_name = "Paul/hatecheck-mandarin"
    elif lang == 'es': # gender
        dataset_name = "Paul/hatecheck-spanish" 
    elif lang == 'pt': #
        dataset_name = "Paul/hatecheck-portuguese"
    elif lang == 'de':
        dataset_name = "Paul/hatecheck-german"
    elif lang == 'it':
        dataset_name = "Paul/hatecheck-italian"
    elif lang == 'ar':
        dataset_name = "Paul/hatecheck-arabic"
    elif lang == 'du':
        dataset_name = "Paul/hatecheck-dutch"
    elif lang == 'fr':
        dataset_name = "Paul/hatecheck-french"
    elif lang == 'po':
        dataset_name = "Paul/hatecheck-polish"
    elif lang == 'hi':
        dataset_name = "Paul/hatecheck-hindi"

    else: dataset_name = "Paul/hatecheck"
    return dataset_name

dataset_name = get_dataset_name(lang)
dataset = load_dataset(dataset_name, split="test")  #, split="test"


case_templ_list = []
non_case_templ_list = []
idt_list = []

for i in dataset:
    # get hate template and idt
    if i['label_gold'] == 'hateful':
        idt_list.append(i['target_ident'])
        #hate_idt_list.append(i['test_case'])
        if lang == 'es' or lang == 'it':
            case_templ_list.append(i['gender_female'])
            case_templ_list.append(i['gender_male'])
        else: case_templ_list.append(i['case_templ'])

    ## non hate template
    else:
        if i['case_templ'] != None:
            if lang == 'en': 
                if '[IDENTITY_P]' in i['case_templ']: non_case_templ_list.append(i['case_templ'])
                else: pass
            elif '[IDENT_P]' in i['case_templ']: non_case_templ_list.append(i['case_templ'])
            else: print(' NON hate speech of NON english, and NON template')
        elif i['gender_male'] != None:
            if '[male_IDENT_P]' in i['gender_male']: non_case_templ_list.append(i['gender_male'])
        elif i['gender_female'] != None:
            if '[female_IDENT_P]' in i['gender_female']: non_case_templ_list.append(i['gender_female'])
        else: pass
        
non_case_templ_list = list(set(non_case_templ_list))
non_case_templ_list = [x for x in non_case_templ_list if x is not None]
case_templ_list = list(set(case_templ_list))
case_templ_list = [x for x in case_templ_list if x is not None]
idt_list = list(set(idt_list))
idt_list = [x for x in idt_list if x is not None]


print('length of nonhate template and hate template: ', len(non_case_templ_list), len(case_templ_list))
corpus_num = min(len(non_case_templ_list), len(case_templ_list))


#### get translated idt
translated_idt_list = []
for idt in idt_list:
    print(idt)
    if lang == 'zh': 
        pass
    else: 
        if lang == 'du': translated_idt = GoogleTranslator(source='auto', target='nl').translate(idt) 
        elif lang == 'po': translated_idt = GoogleTranslator(source='auto', target='pl').translate(idt) 
        else: translated_idt = GoogleTranslator(source='auto', target=lang).translate(idt) 
        print(translated_idt)
        translated_idt_list.append(translated_idt)

if lang == 'zh':
    translated_idt_list = ['穆斯林', '残障人', '变性人', '女人', '黑人', '同性恋', '外国人']
elif lang == 'hi':
    translated_idt_list = ['Musalmaanon', 'kaale log', 'aurat', 'nichli jaati', 'uttar-poorvi bhartiya', 'hijada', 'samlaingik log']
##########

idt_filename = f'idt_list.csv'
with open(idt_filename, 'a') as f:    
    f.write(str(lang)+',')
    f.write(idt_list+',')
    f.write(translated_idt_list)
    f.write('\n')


case_templ_name = f'{lang}/hate_templ.json'
with open(case_templ_name, 'w') as f:
    json.dump(case_templ_list, f, indent=4, ensure_ascii=False) 


non_case_templ_name = f'{lang}/nonhate_templ.json'
with open(non_case_templ_name, 'w') as f:
    json.dump(non_case_templ_list, f, indent=4, ensure_ascii=False) 