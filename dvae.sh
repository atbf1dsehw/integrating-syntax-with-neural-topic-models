#!/bin/bash

data_set_names=( "20ng" "yelp" "imdb" "ag_news" "rotten_tomatoes" "amazon_polarity" "govreport-summarization" )
model_names=( "dvae" )
pre_process=( "true" )
device="cuda:5"
num_topics=( 50 )
save_path="/raid/work/username/final_results/50topics"

for data_set_name in "${data_set_names[@]}"
do
    for model_name in "${model_names[@]}"
    do
        for pre in "${pre_process[@]}"
        do
            for num_topic in "${num_topics[@]}"
            do
            python -m src.main --data_name $data_set_name --model_name $model_name --preprocess $pre --device $device --save_dir $save_path --num_topics $num_topic
            done
        done
    done
done

model_names=( "dvae" )
pre_process=( "true" )
device="cuda:5"
num_topics=( 200 )
save_path="/raid/work/username/final_results/200topics"

for data_set_name in "${data_set_names[@]}"
do
    for model_name in "${model_names[@]}"
    do
        for pre in "${pre_process[@]}"
        do
            for num_topic in "${num_topics[@]}"
            do
            python -m src.main --data_name $data_set_name --model_name $model_name --preprocess $pre --device $device --save_dir $save_path --num_topics $num_topic
            done
        done
    done
done