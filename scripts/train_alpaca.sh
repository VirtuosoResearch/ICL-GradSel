hf_key="meta-llama/Llama-3.2-1B"
sv="fast_approximation_gradients_double_adapter_quant2"
bs=8
dev=1
rf=128
# "EleutherAI/gpt-j-6b"
# "meta-llama/Llama-3.2-1B"
# "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
# "meta-llama/CodeLlama-7b-hf"
# "openai-community/gpt2-medium"
# "meta-llama/Llama-2-13b-hf"
# "meta-llama/CodeLlama-34b-hf"

python train_alpaca.py \
    --model_key $hf_key --lora_rank 4 --lora_alpha 32 --precision "bf16-true"\
    --batch_size $bs --max_length 256 \
    --devices $dev --strategy auto --compute_pretrained_outputs --save_name $sv\
    --downsample 1000 --num_batches_gradients 1000\
    --train_adapter --reduction_factor $rf --epochs 50 --loss_file "loss_1k_50_00005.log" --num_samples 20 --lr 5e-5