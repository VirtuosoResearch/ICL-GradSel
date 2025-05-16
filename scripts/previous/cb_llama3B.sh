
# ------------------topk---------------------
python test.py --dataset superglue-cb --gpt2 meta-llama/Llama-3.2-3B --method direct --out_dir out/meta-llama-Llama-3.2-3B --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 1 --topk

python test.py --dataset superglue-cb --gpt2 meta-llama/Llama-3.2-3B --method direct --out_dir out/meta-llama-Llama-3.2-3B --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 2 --topk

python test.py --dataset superglue-cb --gpt2 meta-llama/Llama-3.2-3B --method direct --out_dir out/meta-llama-Llama-3.2-3B --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 3 --topk

python test.py --dataset superglue-cb --gpt2 meta-llama/Llama-3.2-3B --method direct --out_dir out/meta-llama-Llama-3.2-3B --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 4 --topk

python test.py --dataset superglue-cb --gpt2 meta-llama/Llama-3.2-3B --method direct --out_dir out/meta-llama-Llama-3.2-3B --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 5 --topk

python test.py --dataset superglue-cb --gpt2 meta-llama/Llama-3.2-3B --method direct --out_dir out/meta-llama-Llama-3.2-3B --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 6 --topk


# ------------------supcon---------------------

python test.py --dataset superglue-cb --gpt2 meta-llama/Llama-3.2-3B --method direct --out_dir out/meta-llama-Llama-3.2-3B --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 10 --supcon --m 2

python test.py --dataset superglue-cb --gpt2 meta-llama/Llama-3.2-3B --method direct --out_dir out/meta-llama-Llama-3.2-3B --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 10 --supcon --m 3

python test.py --dataset superglue-cb --gpt2 meta-llama/Llama-3.2-3B --method direct --out_dir out/meta-llama-Llama-3.2-3B --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 10 --supcon --m 4

python test.py --dataset superglue-cb --gpt2 meta-llama/Llama-3.2-3B --method direct --out_dir out/meta-llama-Llama-3.2-3B --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 10 --supcon --m 5

python test.py --dataset superglue-cb --gpt2 meta-llama/Llama-3.2-3B --method direct --out_dir out/meta-llama-Llama-3.2-3B --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 10 --supcon --m 6



python test.py --dataset superglue-cb --gpt2 meta-llama/Llama-3.2-3B --method direct --out_dir out/meta-llama-Llama-3.2-3B --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 15 --supcon --m 2

python test.py --dataset superglue-cb --gpt2 meta-llama/Llama-3.2-3B --method direct --out_dir out/meta-llama-Llama-3.2-3B --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 15 --supcon --m 3

python test.py --dataset superglue-cb --gpt2 meta-llama/Llama-3.2-3B --method direct --out_dir out/meta-llama-Llama-3.2-3B --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 15 --supcon --m 4

python test.py --dataset superglue-cb --gpt2 meta-llama/Llama-3.2-3B --method direct --out_dir out/meta-llama-Llama-3.2-3B --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 15 --supcon --m 5

python test.py --dataset superglue-cb --gpt2 meta-llama/Llama-3.2-3B --method direct --out_dir out/meta-llama-Llama-3.2-3B --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 15 --supcon --m 6

# ------------------randomk---------------------

python test.py --dataset superglue-cb --gpt2 meta-llama/Llama-3.2-3B --method direct --out_dir out/meta-llama-Llama-3.2-3B --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 1 --randomk

python test.py --dataset superglue-cb --gpt2 meta-llama/Llama-3.2-3B --method direct --out_dir out/meta-llama-Llama-3.2-3B --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 2 --randomk

python test.py --dataset superglue-cb --gpt2 meta-llama/Llama-3.2-3B --method direct --out_dir out/meta-llama-Llama-3.2-3B --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 3 --randomk

python test.py --dataset superglue-cb --gpt2 meta-llama/Llama-3.2-3B --method direct --out_dir out/meta-llama-Llama-3.2-3B --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 4 --randomk

python test.py --dataset superglue-cb --gpt2 meta-llama/Llama-3.2-3B --method direct --out_dir out/meta-llama-Llama-3.2-3B --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 5 --randomk

python test.py --dataset superglue-cb --gpt2 meta-llama/Llama-3.2-3B --method direct --out_dir out/meta-llama-Llama-3.2-3B --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 6 --randomk
