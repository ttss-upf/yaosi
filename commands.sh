python train_single.py \
  --batch_size 32 --enc_type transformer --dec_type transformer --vocab_size 12360 \
  --emb_dim 512 --hidden_size 512 --filter_size 512 \
  --enc_layers 2 --dec_layers 2 --num_heads 2  \
  --eval_steps 20 --checkpoint 1000 --dropout 0.01 \
  --lang eng --max_len 128 --lr 0.1 --weight_decay 0.01 --epochs 3 \
