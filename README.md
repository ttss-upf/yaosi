#要死


最多字符串在180（106 + 65 ）左右

词典长度12360个


python train_single.py \
  --batch_size 1 --enc_type transformer --dec_type transformer --vocab_size 12122 \
  --emb_dim 512 --hidden_size 512 --filter_size 512 \
  --enc_layers 2 --dec_layers 2 --num_heads 2  \
  --eval_steps 1000 --checkpoint 1000 --dropout 0.2 \
  --lang eng \
