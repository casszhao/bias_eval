#!/bin/bash


filename="results/idtB"
date=$(date +"%m%d_%H%M")
extension=".csv"

# Concatenate variables to create filename
new_filename="${filename}_${date}${extension}"

# Create new file with concatenated filename
echo "Language,Corpus_Size,Monolingual,Mono_Pvalue,Multilingual,Multi_Pvalue,Diff_in_Scores,MonoModel_Size,MultiModel_Size,Mono_token_len,Multi_token_len" > "$new_filename"



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