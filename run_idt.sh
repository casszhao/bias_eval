#!/bin/bash

echo '  ' >> results/idt.txt
echo '  ' >> results/idt.txt
echo '  ' >> results/idt.txt
echo $(date +%F-%T) >> results/idt.txt



echo '  ' >> results/idt.txt
echo '  ' >> results/idt.txt
lan='zh'
echo $lan >> results/idt.txt
python eval_bias.py --lang $lan --if_multilingual 'multi'
python eval_bias.py --lang $lan --if_multilingual 'mono'


echo '  ' >> results/idt.txt
echo '  ' >> results/idt.txt
lan='du'
echo $lan >> results/idt.txt
python eval_bias.py --lang $lan --if_multilingual 'multi'
python eval_bias.py --lang $lan --if_multilingual 'mono'


echo '  ' >> results/idt.txt
echo '  ' >> results/idt.txt
lan='fr'
echo $lan >> results/idt.txt
python eval_bias.py --lang $lan --if_multilingual 'multi'
python eval_bias.py --lang $lan --if_multilingual 'mono'


echo '  ' >> results/idt.txt
echo '  ' >> results/idt.txt
lan='po'
echo $lan >> results/idt.txt
python eval_bias.py --lang $lan --if_multilingual 'multi'
python eval_bias.py --lang $lan --if_multilingual 'mono'
