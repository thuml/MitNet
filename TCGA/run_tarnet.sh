export CUDA_VISIBLE_DEVICES=0

python train.py \
  --path ../data \
  --model TARNet \
  --batch-size 32 \
  --lr 0.0001 \
  --wd 0.005 \
  --log TARNet