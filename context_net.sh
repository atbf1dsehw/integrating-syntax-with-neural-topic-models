#!/bin/bash

data_set_names=( "20ng" "yelp" "imdb" "ag_news" "rotten_tomatoes" "amazon_polarity" "govreport-summarization" )
model_names=( "lm" )
pre_process=( "false" )
context_size=( 1 2 3 4 5 )
context_type=( "symmetric" )
device="cuda:4"
save_path="/raid/work/username/final_results/lm_results"

for data_set_name in "${data_set_names[@]}"
do
    for model_name in "${model_names[@]}"
    do
        for pre in "${pre_process[@]}"
        do
            for context in "${context_size[@]}"
            do
                for context_t in "${context_type[@]}"
                do
                    python -m src.main --data_name $data_set_name --model_name $model_name --preprocess $pre --device $device --save_dir $save_path --context_size $context --context_type $context_t
                done
            done
        done
    done
done


data_set_names=( "20ng" "yelp" "imdb" "ag_news" "rotten_tomatoes" "amazon_polarity" "govreport-summarization" )
model_names=( "lm" )
pre_process=( "false" )
context_size=( 5 )
context_type=( "asymmetric" )
device="cuda:4"

for data_set_name in "${data_set_names[@]}"
do
    for model_name in "${model_names[@]}"
    do
        for pre in "${pre_process[@]}"
        do
            for context in "${context_size[@]}"
            do
                for context_t in "${context_type[@]}"
                do
                    python -m src.main --data_name $data_set_name --model_name $model_name --preprocess $pre --device $device --save_dir $save_path --context_size $context --context_type $context_t
                done
            done
        done
    done
done