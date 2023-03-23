#!/bin/bash

echo '  ' >> results/gender.txt
echo '  ' >> results/gender.txt
echo '  ' >> results/gender.txt
echo $(date +%F-%T) >> results/gender.txt



echo '  ' >> results/gender.txt
echo '  ' >> results/gender.txt
lan='de'
echo $lan >> results/gender.txt
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'mono'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'mono'



echo '  ' >> results/gender.txt
echo '  ' >> results/gender.txt
lan='ja'
echo $lan >> results/gender.txt
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'mono'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'mono'



echo '  ' >> results/gender.txt
echo '  ' >> results/gender.txt
lan='ar'
echo $lan >> results/gender.txt
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'mono'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'mono'



echo '  ' >> results/gender.txt
echo '  ' >> results/gender.txt
lan='es'
echo $lan >> results/gender.txt
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'mono'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'mono'




echo '  ' >> results/gender.txt
echo '  ' >> results/gender.txt
lan='pt'
echo $lan >> results/gender.txt
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'mono'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'mono'





echo '  ' >> results/gender.txt
echo '  ' >> results/gender.txt
lan='ru'
echo $lan >> results/gender.txt
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'mono'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'mono'




echo '  ' >> results/gender.txt
echo '  ' >> results/gender.txt
lan='id'
echo $lan >> results/gender.txt
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'mono'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'mono'




echo '  ' >> results/gender.txt
echo '  ' >> results/gender.txt
lan='zh'
echo $lan >> results/gender.txt
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'ted' --if_multilingual 'mono'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'multi'
python eval_gender.py --lang $lan --corpus 'news' --if_multilingual 'mono'