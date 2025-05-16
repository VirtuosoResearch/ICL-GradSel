# python test.py --dataset poem_sentiment --gpt2  deepseek-ai/deepseek-llm-7b-chat --method direct --out_dir out/Llama-3.2-1B --do_zeroshot --test_batch_size 4 --use_demonstrations --seed 100 --k 3 --ground --is_quant

# python test.py --dataset climate_fever --gpt2  deepseek-ai/deepseek-llm-7b-chat --method direct --out_dir out/Llama-3.2-1B --do_zeroshot --test_batch_size 4 --use_demonstrations --seed 100 --k 3 --ground  --is_quant

# python test.py --dataset medical_questions_pairs --gpt2  deepseek-ai/deepseek-llm-7b-chat --method direct --out_dir out/Llama-3.2-1B --do_zeroshot --test_batch_size 4 --use_demonstrations --seed 100 --k 3 --ground  --is_quant

# python test.py --dataset glue-rte --gpt2  deepseek-ai/deepseek-llm-7b-chat --method direct --out_dir out/Llama-3.2-1B --do_zeroshot --test_batch_size 4 --use_demonstrations --seed 100 --k 3 --ground  --is_quant

# python test.py --dataset strategyqa --gpt2  deepseek-ai/deepseek-llm-7b-chat --method direct --out_dir out/Llama-3.2-1B --do_zeroshot --test_batch_size 4 --use_demonstrations --seed 100 --k 3 --ground  --is_quant

python test.py --dataset medical_questions_pairs --gpt2  deepseek-ai/deepseek-llm-7b-chat --method direct --out_dir out/Llama-3.2-1B --do_zeroshot --test_batch_size 4 --use_demonstrations --seed 100 --k 3 --ground  --is_quant --is_flops
