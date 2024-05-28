# supports distilbert-base-cased, roberta-large, gpt2-xl, facebook/opt-2.7b
# algorithm list - {'FO', 'ZO', 'ZOSVRG'}
# tasks supported - {'mnli', 'sst2', 'qnli', 'cola'}

gpu=$1
task=$2
alg=$3
an=$4
lr=$5
lr_mb=$6
PARTIAL=$7
EXP_PATH=$8
LAST=$9
SOFT_PROMPT=$10

ADD_ARGS=''

if [[ "$PARTIAL" == "full" ]]; then
    ADD_ARGS+='--full_parameter '
fi

if [[ "$SOFT_PROMPT" == "y" ]]; then
    ADD_ARGS+='--soft_prompt '
fi

if [[ "$alg" == "ZO" ]]; then
    epoch=32000
elif [[ "$alg" == "ZOSVRG" ]]; then
    epoch=8000
fi

bs=64
# rm -r ~/.cache/huggingface/datasets/*

logarg=gpu${gpu}_task${task}_alg${alg}_an${an}_lr${lr}_lr_mb${lr_mb}_${PARTIAL}

ALL_ARGS="
    --epochs $epoch \
    --samplesize 512 \
    --samplesize_validation 256 \
    --model_name roberta-large \
    --task $task \
    --max_seq_length 128 \
    $ADD_ARGS \
    --algorithm $alg \
    --q 2 \
    --batchsize $bs \
    --batchsize_limit 64 \
    --anneal $an \
    --lr $lr \
    --lr_mezosvrg_mb $lr_mb \
    --device $gpu \
    --results $EXP_PATH/results_$logarg 
"

echo $ALL_ARGS

if [[ "$LAST" == "t" ]]; then
    python /fsx/users/zhuha/ZO_SmallScaleExp/finetune_llm.py $ALL_ARGS >& $EXP_PATH/log_$logarg.log
elif [[ "$LAST" == "f" ]]; then
    python /fsx/users/zhuha/ZO_SmallScaleExp/finetune_llm.py $ALL_ARGS >& $EXP_PATH/log_$logarg.log &
fi