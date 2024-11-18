MODEL=facebook/opt-13b 
task_names=( "RTE" "CB" "SST2" "BoolQ" "WSC" "WIC" "MultiRC" "Copa" "ReCoRD" "SQuAD" "DROP")
# "RTE" "CB" "SST2" "BoolQ" "WSC" "WIC" "MultiRC" "Copa" "ReCoRD" "SQuAD" "DROP"
MODEL=${MODEL:-facebook/opt-13b}
length=${#task_names[@]}

for ((j = 0; j < $length; j++)); do
    python run.py --model_name $MODEL --task_name ${task_names[$j]} --output_dir result/tmp --tag icl --num_train 32 --num_eval 1000 --load_float16 --verbose False
done