#!/bin/bash


filename="results/subgroup"
date=$(date +"%m%d_%H%M")
extension=".csv"

# Concatenate variables to create filename
new_filename="${filename}_${date}${extension}"

# Create new file with concatenated filename
echo "Language;Mono_MUlti;bias_group;disadv_corpus;bias_score;p-value;most_biasedpair_hate_idt;most_biasedpair_disadv;token_num;corpus_num" > "$new_filename"




python eval_bias_subgroup.py --lang 'de' --save_results $new_filename
python eval_bias_subgroup.py --lang 'du' --save_results $new_filename
python eval_bias_subgroup.py --lang 'en' --save_results $new_filename
python eval_bias_subgroup.py --lang 'es' --save_results $new_filename
python eval_bias_subgroup.py --lang 'fr' --save_results $new_filename
python eval_bias_subgroup.py --lang 'hi' --save_results $new_filename
python eval_bias_subgroup.py --lang 'it' --save_results $new_filename
python eval_bias_subgroup.py --lang 'po' --save_results $new_filename
python eval_bias_subgroup.py --lang 'pt' --save_results $new_filename
python eval_bias_subgroup.py --lang 'zh' --save_results $new_filename
python eval_bias_subgroup.py --lang 'ar' --save_results $new_filename