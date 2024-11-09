

#################################################### cat token L 2-10 #####################################################
TORCH_DISTRIBUTED_DEBUG=INFO torchrun --nnodes=1 --nproc_per_node=4 sample_fid.py \
 --model GPT-L --modeling mlm --sample_dir images_fid_mlm \
 --ckpt /model_home/mlmL-1-16.pth \
 --dataset custom --codebook_size 16 --norm_first --num-classes 1000 \
 --code-dim 16 --token-each 1 --pos_type rope2d --cfg-scale 3.0 --cfg_schedule linear \
 --deterministic --gen_iter_num 256 --temperature 9.0 \
 --num_images 32000 --per_proc_batch_size 16 --global_seed 64

#python -m pytorch_fid /app/ms/AIGC/haoshaozhe/fid_test/pytorch_fid/imagenet/train.npz images_fid_ar/L-2-10-g256-linearcfg-4.0-t1.0-g256-top1000 --num-workers 4
