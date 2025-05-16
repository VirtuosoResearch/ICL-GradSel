# python test.py --dataset superglue-cb --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 7 --topk

# python test.py --dataset superglue-cb --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 7 --randomk

# python test.py --dataset superglue-cb --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 14 --unlabeled

# python test.py --dataset superglue-cb --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 7 --randomk

python test.py --dataset superglue-cb --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 10 --supcon --m 7

python test.py --dataset superglue-cb --gpt2 gpt2-large --method direct --out_dir out/gpt2-large --do_zeroshot --test_batch_size 4 --use_demonstrations  --seed 100 --k 15 --supcon --m 7
