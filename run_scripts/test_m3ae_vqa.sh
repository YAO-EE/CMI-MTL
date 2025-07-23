num_gpus=1
per_gpu_batchsize=16

# === 1. VQA ===
# === VQA-RAD ===
# === Provided Checkpoints ===
# downloaded/finetuned/vqa/vqa_rad/m3ae_finetuned_vqa_vqa_rad_77.4.ckpt
# downloaded/finetuned/vqa/vqa_rad/m3ae_finetuned_vqa_vqa_rad_76.9.ckpt
# downloaded/finetuned/vqa/vqa_rad/m3ae_finetuned_vqa_vqa_rad_76.7.ckpt
python main.py with data_root=data/finetune_arrows/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_vqa_vqa_rad \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 \
 test_only=True \
 tokenizer=/mnt/3bb01a96-4d48-4134-9dcc-1e1cdd11daa3/zxy/M3AE-master/downloaded/roberta-base \
 load_path=/mnt/3bb01a96-4d48-4134-9dcc-1e1cdd11daa3/zxy/M3AE-master/downloaded/finetuned/vqa/vqa_rad/m3ae_finetuned_vqa_vqa_rad_77.4.ckpt

# === SLACK ===
# === Provided Checkpoints ===
# downloaded/finetuned/vqa/slack/m3ae_finetuned_vqa_slack_84.3.ckpt
# downloaded/finetuned/vqa/slack/m3ae_finetuned_vqa_slack_83.0.ckpt
# downloaded/finetuned/vqa/slack/m3ae_finetuned_vqa_slack_82.5.ckpt
python main.py with data_root=data/finetune_arrows/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_vqa_slack \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 \
 test_only=True \
 tokenizer=/mnt/3bb01a96-4d48-4134-9dcc-1e1cdd11daa3/zxy/M3AE-master/downloaded/roberta-base \
 load_path=/mnt/3bb01a96-4d48-4134-9dcc-1e1cdd11daa3/zxy/M3AE-master/downloaded/finetuned/vqa/slack/m3ae_finetuned_vqa_slack_84.3.ckpt

# === MedVQA-2019 ===
# === Provided Checkpoints ===
# downloaded/finetuned/vqa/medvqa_2019/m3ae_finetuned_vqa_medvqa_2019_80.5.ckpt
# downloaded/finetuned/vqa/medvqa_2019/m3ae_finetuned_vqa_medvqa_2019_80.0.ckpt
# downloaded/finetuned/vqa/medvqa_2019/m3ae_finetuned_vqa_medvqa_2019_79.7.ckpt
python main.py with data_root=data/finetune_arrows/ \
 num_gpus=${num_gpus} num_nodes=1 \
 task_finetune_vqa_medvqa_2019 \
 per_gpu_batchsize=${per_gpu_batchsize} \
 clip16 text_roberta \
 image_size=384 \
 test_only=True \
 tokenizer=/mnt/3bb01a96-4d48-4134-9dcc-1e1cdd11daa3/zxy/M3AE-master/downloaded/roberta-base \
 load_path=/mnt/3bb01a96-4d48-4134-9dcc-1e1cdd11daa3/zxy/M3AE-master/downloaded/finetuned/vqa/medvqa_2019/m3ae_finetuned_vqa_medvqa_2019_80.5.ckpt