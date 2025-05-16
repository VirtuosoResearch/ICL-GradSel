python test.py --dataset poem_sentiment --gpt2 meta-llama/Llama-3.1-8B --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations --seed 100 --k 3 --estim --device 0 --is_quant --pseudo_k 3

python test.py --dataset poem_sentiment --gpt2 meta-llama/Llama-3.1-8B --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations --seed 100 --k 3 --estim --device 0 --is_quant --pseudo_k 4

python test.py --dataset poem_sentiment --gpt2 meta-llama/Llama-3.1-8B --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations --seed 100 --k 3 --estim --device 0 --is_quant --pseudo_k 5

python test.py --dataset poem_sentiment --gpt2 meta-llama/Llama-3.1-8B --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations --seed 100 --k 3 --estim --device 0 --is_quant --pseudo_k 6