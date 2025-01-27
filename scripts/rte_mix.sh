
# ------------------topk---------------------
python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 1 --topk

python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 2 --topk

python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 3 --topk

python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 4 --topk

python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 5 --topk

python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 6 --topk


# ------------------supcon---------------------

python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 10 --supcon --m 2

python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 10 --supcon --m 3

python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 10 --supcon --m 4

python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 10 --supcon --m 5

python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 10 --supcon --m 6




python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 15 --supcon --m 2

python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 15 --supcon --m 3

python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 15 --supcon --m 4

python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 15 --supcon --m 5

python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 15 --supcon --m 6




# # ------------------randomk---------------------

# python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 1 --randomk

# python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 2 --randomk

# python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 3 --randomk

# python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 4 --randomk

# python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 5 --randomk

# python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 6 --randomk

# python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 7 --randomk

# # --------------------unlabeled--------------------

# python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 2 --unlabeled

# python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 4 --unlabeled

# python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 6 --unlabeled

# python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 8 --unlabeled

# python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 10 --unlabeled

# python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 12 --unlabeled

# python test.py --dataset glue-rte --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 14 --unlabeled
