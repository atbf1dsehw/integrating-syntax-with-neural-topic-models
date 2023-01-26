#!/bin/bash

data_set_names=( "yelp" "20ng" "imdb" "ag_news" "rotten_tomatoes" "amazon_polarity" "govreport-summarization" )
model_names=( "syconntm" )
pre_process=( "false" )
lambda=( 0.5 )
context_size=( 5 )
context_type=( "symmetric" )
device="cuda:1"
save_path="/raid/work/username/final_results/qualitative_results"
topics=10
num_syn_topics=10
top_n=5

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
                    for l in "${lambda[@]}"
                    do
                        python -m src.main --data_name $data_set_name --model_name $model_name --preprocess $pre --device $device --save_dir $save_path --context_size $context --context_type $context_t --lambda_ $l --num_topics $topics --num_syn_topics $num_syn_topics --top_n $top_n
                    done
                done
            done
        done
    done
done


data_set_names=( "yelp" "20ng" "imdb" "ag_news" "rotten_tomatoes" "amazon_polarity" "govreport-summarization" )
model_names=( "dvae" "etm" "etm_dirichlet" "lda" )
pre_process=( "true" "false" )

for data_set_name in "${data_set_names[@]}"
do
    for model_name in "${model_names[@]}"
    do
        for pre in "${pre_process[@]}"
        do
            python -m src.main --data_name $data_set_name --model_name $model_name --preprocess $pre --device $device --save_dir $save_path --num_topics $topics --num_syn_topics $num_syn_topics
        done
    done
done