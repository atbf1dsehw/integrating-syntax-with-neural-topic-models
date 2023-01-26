#!/bin/bash

data_set_names=( "20ng" "yelp" "imdb" "ag_news" "rotten_tomatoes" "amazon_polarity" "govreport-summarization" )
model_names=( "syconntm" )
pre_process=( "false" )
lambda=( 0.5 )
context_size=( 5 )
context_type=( "symmetric" )
device="cuda:2"
save_path="/raid/work/username/final_results/50topics"
num_topics=( 50 )
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
                        for t in "${num_topics[@]}"
                        do
                        python -m src.main --data_name $data_set_name --model_name $model_name --preprocess $pre --device $device --save_dir $save_path --context_size $context --context_type $context_t --lambda_ $l --num_topics $t --top_n $top_n
                        done
                    done
                done
            done
        done
    done
done


data_set_names=( "20ng" "yelp" "imdb" "ag_news" "rotten_tomatoes" "amazon_polarity" "govreport-summarization" )
model_names=( "syconntm" )
pre_process=( "false" )
lambda=( 0.5 )
context_size=( 5 )
context_type=( "symmetric" )
device="cuda:2"
num_topics=( 200 )
save_path="/raid/work/username/final_results/200topics"

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
                        for t in "${num_topics[@]}"
                        do
                            python -m src.main --data_name $data_set_name --model_name $model_name --preprocess $pre --device $device --save_dir $save_path --context_size $context --context_type $context_t --lambda_ $l --num_topics $t --top_n $top_n
                        done
                    done
                done
            done
        done
    done
done