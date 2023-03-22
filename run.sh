#!/bin/bash

echo $(date +%F-%T) >> results/gender.txt

lan='ja'
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'mono'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'mono'

lan='en'
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'mono'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'mono'

lan='it'
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'mono'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'mono'

lan='de'
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'mono'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'mono'

lan='pt'
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'mono'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'mono'

lan='es'
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'multi'
python 
eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'mono'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'mono'