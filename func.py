


def get_model_name_cased(lang_model):
    if lang_model == 'de-bert':
        model_name =  "dbmdz/bert-base-german-cased" #"TurkuNLP/wikibert-base-de-cased" # 'deepset/gbert-base' 
    elif lang_model == 'de-distilbert':
        model_name ='distilbert-base-german-cased'


    elif lang_model == 'hi-bert':
        model_name = 'l3cube-pune/hindi-bert-scratch'

    elif lang_model == 'en-bert':
        model_name = 'bert-base-cased'
    elif lang_model == 'en-deberta':
        model_name = 'microsoft/deberta-v3-base'
    elif lang_model == 'en-distilbert':
        model_name = 'distilbert-base-cased'
    elif lang_model == 'en-roberta':
        model_name = 'roberta-base'
        
    elif lang_model == 'it-bert':
        model_name = "dbmdz/bert-base-italian-cased" # "dbmdz/bert-base-italian-uncased"
    elif lang_model == 'it-xlm':
        model_name = "MilaNLProc/hate-ita-xlm-r-base"
    
    elif lang_model == 'es-bert': 
        model_name = "dccuchile/bert-base-spanish-wwm-cased" #'dccuchile/bert-base-spanish-wwm-uncased'

    elif lang_model == 'pt-bert': 
        model_name = "neuralmind/bert-large-portuguese-cased" # pablocosta/bertabaporu-base-uncased neuralmind/bert-base-portuguese-cased 
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
        model_name = 'aubmindlab/bert-base-arabert' 

    elif lang_model == 'id-bert':
        model_name = 'cahya/bert-base-indonesian-1.5G'

    elif lang_model == 'zh-bert':
        model_name = 'bert-base-chinese'

    elif lang_model == 'du-bert':
        model_name = 'GroNLP/bert-base-dutch-cased'
    elif lang_model == 'fr-bert':
        model_name = 'dbmdz/bert-base-french-europeana-cased'

    elif lang_model == 'po-bert':
        model_name = 'dkleczek/bert-base-polish-uncased-v1'


    elif lang_model == 'multi-bert':
        model_name = 'bert-base-multilingual-cased' # bert-base-multilingual-uncased
    elif lang_model == 'multi-deberta':
        model_name = 'microsoft/mdeberta-v3-base' # large 
    elif lang_model == 'multi-distilbert':
        model_name ='distilbert-base-multilingual-cased'
    return model_name

def get_model_name_uncased(lang_model):
    if lang_model == 'de-bert':
        model_name = 'deepset/gbert-base' # s
    elif lang_model == 'de-distilbert':
        model_name ='distilbert-base-german-cased'

    elif lang_model == 'du-bert':
        model_name = 'GroNLP/bert-base-dutch-cased'
    elif lang_model == 'fr-bert':
        model_name = 'dbmdz/bert-base-french-europeana-cased'

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
        model_name = 'dccuchile/bert-base-spanish-wwm-uncased' # s


    elif lang_model == 'po-bert': 
        model_name = 'dkleczek/bert-base-polish-uncased-v1'

    elif lang_model == 'pt-bert': 
        model_name = "pablocosta/bertabaporu-base-uncased" # ns neuralmind/bert-base-portuguese-cased 
    elif lang_model == 'pt-xlm':
        model_name = "thegoodfellas/tgf-xlm-roberta-base-pt-br" 

    elif lang_model == 'ar-bert': 
        model_name = 'aubmindlab/bert-base-arabertv02' # s


    elif lang_model == 'ru-bert': 
        model_name = 'blinoff/roberta-base-russian-v0' # s
    elif lang_model == 'ru-distilbert': 
        model_name = 'Geotrend/distilbert-base-ru-cased'

    elif lang_model == 'ja-bert': 
        model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking' # s
    elif lang_model == 'ja-distilbert': 
        model_name = 'laboro-ai/distilbert-base-japanese'



    elif lang_model == 'id-bert':
        model_name = 'cahya/bert-base-indonesian-1.5G' # s

    elif lang_model == 'zh-bert':
        model_name = 'hfl/chinese-bert-wwm-ext' # s


    elif lang_model == 'multi-bert':
        model_name = 'bert-base-multilingual-uncased' # bert-base-multilingual-uncased
    elif lang_model == 'multi-deberta':
        model_name = 'microsoft/mdeberta-v3-base' # large 
    elif lang_model == 'multi-distilbert':
        model_name ='distilbert-base-multilingual-cased'
    return model_name

