# M3AE

This is the official implementation of [CMI-MTL: Cross-Mamba interaction based multi-task learning for medical visual question answering] at PG-2025.

## Table of Contents
- [Requirements](#requirements)
- [Download](#download)
- [Downstream Evaluation](#downstream-evaluation)

## Requirements
Run the following command to install the required packages:
```bash
pip install -r requirements.txt
```

## Download
You can download the models we pre-trained and fine-tuned in the corresponding datasets from [here]().

## Downstream Evaluation
### 1. Dataset Preparation
Please organize the fine-tuning datasets as the following structure:
```angular2
root:[data]
+--finetune_data
| +--slack
| | +--train.json
| | +--validate.json
| | +--test.json
| | +--imgs
| +--vqa_rad
| | +--trainset.json
| | +--valset.json
| | +--testset.json
| | +--images
| +--ovqa
| | +--val
| | +--test
| | +--train
```

### 2. Pre-processing
Run the following command to pre-process the data:
```angular2
python prepro/prepro_finetuning_data.py
```
to get the following arrow files:
```angular2
root:[data]
+--finetune_arrows
| +--vqa_vqa_rad_train.arrow
| +--vqa_vqa_rad_val.arrow
| +--vqa_vqa_rad_test.arrow
| +--vqa_slack_train.arrow
| +--vqa_slack_test.arrow
| +--vqa_slack_val.arrow
| +--vqa_ovqa_train.arrow
| +--vqa_ovqa_2019_val.arrow
| +--vqa_ovqa_2019_test.arrow
```

### 3. Fine-Tuning
Now you can start to fine-tune the m3ae model:
```angular2
bash run_scripts/finetune_m3ae.sh
```

### 4. Test
You can also test our fine-tuned models directly:
```angular2
bash run_scripts/test_m3ae.sh
```
NOTE: This is a good way to check whether your environment is set up in the same way as ours (if you can reproduce the same results).

## Acknowledgement
The code is based on [M3AE]([https://github.com/zhjohnchan/M3AE]).
We thank the authors for their open-sourced code and encourage users to cite their works when applicable.

## Citations

