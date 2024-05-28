#!/bin/sh
# EXP_PATH=/fsx/users/zhuha/ZO_SmallScaleExp/logs/$(date "+%y-%m-%d-%H-%M-%S")-robertaL-mezo
# mkdir -p $EXP_PATH
# echo $EXP_PATH

# sbatch -N 1\
#        --output=$EXP_PATH/test_output.txt \
#        --error=$EXP_PATH/test_error.txt \
#        --job-name=llama \
#        --partition=benchmark \
#        --wrap "srun /fsx/users/zhuha/ZO_SmallScaleExp/run_mnli_mezo.sh"

# EXP_PATH=/fsx/users/zhuha/ZO_SmallScaleExp/logs/$(date "+%y-%m-%d-%H-%M-%S")-robertaL-svrg-full
# mkdir -p $EXP_PATH
# echo $EXP_PATH

# sbatch -N 1\
#        --output=$EXP_PATH/test_output.txt \
#        --error=$EXP_PATH/test_error.txt \
#        --job-name=llama \
#        --partition=benchmark \
#        --wrap "srun /fsx/users/zhuha/ZO_SmallScaleExp/run_mnli_zosvrg_partial.sh"

# EXP_PATH=/fsx/users/zhuha/ZO_SmallScaleExp/logs/$(date "+%y-%m-%d-%H-%M-%S")-robertaL-svrg-partial
# mkdir -p $EXP_PATH
# echo $EXP_PATH

# sbatch -N 1\
#        --output=$EXP_PATH/test_output.txt \
#        --error=$EXP_PATH/test_error.txt \
#        --job-name=llama \
#        --partition=benchmark \
#        --wrap "srun /fsx/users/zhuha/ZO_SmallScaleExp/run_mnli_zosvrg_full.sh"

EXP_PATH=/fsx/users/zhuha/ZO_SmallScaleExp/logs/$(date "+%y-%m-%d-%H-%M-%S")-robertaL-all
mkdir -p $EXP_PATH
echo $EXP_PATH

sbatch -N 1\
       --output=$EXP_PATH/test_output.txt \
       --error=$EXP_PATH/test_error.txt \
       --job-name=llama \
       --partition=benchmark \
       --wrap "srun /fsx/users/zhuha/ZO_SmallScaleExp/run_mnli_all.sh"