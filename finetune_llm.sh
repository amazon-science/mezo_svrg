# supports distilbert-base-cased, roberta-large, gpt2-xl, facebook/opt-2.7b, facebook/opt-6.7b
# algorithm list - {'FO-SGD', 'FO-Adam', 'ZO', 'ZOSVRG'}
# tasks supported - {'mnli', 'sst2', 'qnli', 'cola'}
# only single gpu experiments are supported. Indicate the device you want run this on
# full_parameter tag does full-parameter fine-tuning. Remove for partial fine-tuning

# For ZO methods, "batchsize" argument is effective batch size after accumulation
# and "batchsize_limit" is true batch size. For FO, ignore batchsize_limit argument
# results argument takes path to store dictionary of results (Losses, Accuracies, Training Time etc.) 
# lr argument is \eta in paper or \eta_1 for MeZO-SVRG
# lr_mezosvrg_mb is \eta_2

python finetune_llm.py \
    --epochs 125 \
    --samplesize 512 \
    --samplesize_validation 256 \
    --model_name 'facebook/opt-1.3b' \
    --full_parameter \
    --task 'mnli' \
    --max_seq_length 2048 \
    --algorithm 'FO-Adam' \
    --q 2 \
    --batchsize 1 \
    --batchsize_limit 1 \
    --anneal 5 \
    --lr 1e-3 \
    --perturbation_scale 1e-3 \
    --lr_mezosvrg_mb 1e-6 \
    --device 0 \
    --half_precision \
    --results 'robertalarge/result_MNLI_RoBERTalarge_PartialParam_SP'