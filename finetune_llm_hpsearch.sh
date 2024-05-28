# supports distilbert-base-cased, roberta-large, gpt2
# algorithm list - {'FO', 'ZO', 'ZOSVRG'}
# tasks supported - {'mnli', 'sst2', 'qnli', 'cola'}

declare -a QArray=(1 2 3 4 5 6 7 8)
DEVICE_COUNTER=0
for q in "${QArray[@]}"; do
    python finetune_llm.py \
        --epochs 1 \
        --samplesize 1024 \
        --samplesize_validation 128 \
        --model_name 'roberta-large' \
        --task 'mnli' \
        --full_parameter \
        --algorithm 'FO' \
        --q $q \
        --batchsize 64 \
        --batchsize_limit 32 \
        --anneal 5 \
        --lr 2e-3 \
        --device $DEVICE_COUNTER \
        --results 'results_demo' &
    let DEVICE_COUNTER=DEVICE_COUNTER+1 
done