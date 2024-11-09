CUDA_VISIBLE_DEVICES="0" python sample.py --modeling mlm --model GPT-L \
 --ckpt /model_home/mlmL-1-16.pth \
 --save_dir output \
 --temperature 9.0 \
 --dataset custom --codebook_size 16 --norm_first \
 --code-dim 16 --token-each 1 --pos_type rope2d --cfg-scale 3.0 --cfg_schedule linear \
 --top-k 400 --gen_iter_num 256 --gen_num 4 \
 --class_labels 985 "$@"