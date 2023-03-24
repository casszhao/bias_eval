#!/bin/bash


filename="results/idt"
date=$(date +"%m%d_%H%M")
extension=".csv"

# Concatenate variables to create filename
new_filename="${filename}_${date}${extension}"

# Create new file with concatenated filename
echo "Language,Corpus Size,Monolingual,Multilingual,Mono_Pvalue,Multi_Pvalue,Diff_in_Scores,MonoModel_Size,MultiModel_Size,Mono_token_len,Multi_token_len" > "$new_filename"



python eval_bias.py --lang 'ar' --log_name $new_filename
python eval_bias.py --lang 'de' --log_name $new_filename
python eval_bias.py --lang 'du' --log_name $new_filename
python eval_bias.py --lang 'en' --log_name $new_filename
python eval_bias.py --lang 'es' --log_name $new_filename
python eval_bias.py --lang 'fr' --log_name $new_filename
python eval_bias.py --lang 'hi' --log_name $new_filename
python eval_bias.py --lang 'it' --log_name $new_filename
python eval_bias.py --lang 'po' --log_name $new_filename
python eval_bias.py --lang 'pt' --log_name $new_filename
python eval_bias.py --lang 'zh' --log_name $new_filename


# echo '  ' >> results/idt.txt
# echo '  ' >> results/idt.txt
# lan='du'
# echo $lan >> results/idt.txt
# python eval_bias.py --lang $lan --if_multilingual 'multi'
# python eval_bias.py --lang $lan --if_multilingual 'mono'


# echo '  ' >> results/idt.txt
# echo '  ' >> results/idt.txt
# lan='fr'
# echo $lan >> results/idt.txt
# python eval_bias.py --lang $lan --if_multilingual 'multi'
# python eval_bias.py --lang $lan --if_multilingual 'mono'


# echo '  ' >> results/idt.txt
# echo '  ' >> results/idt.txt
# lan='po'
# echo $lan >> results/idt.txt
# python eval_bias.py --lang $lan --if_multilingual 'multi'
# python eval_bias.py --lang $lan --if_multilingual 'mono'
