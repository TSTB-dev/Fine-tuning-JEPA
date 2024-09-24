# Fine-tuning-JEPA
Simple I-JEPA implementation for pre-training and fine-tuning your own datasets. 


## Getting Started


### Installation
---

`git clone https://github.com/TSTB-dev/Fine-tuning-JEPA && cd Fine-tuning-JEPA`

`pip install -r requirements.txt`


### Setup datasets & Pre-trained checkpoints
---
For downloading datasets (supported datasets are "stanford-cars", "flowers", "oxford pets", "cub200-2011", "caltech101"). 

`bash scripts/setup_dataset.sh`

For downloading checkpoints (from official)
`bash scripts/download_checkpoints.sh`

### For Pre-Training
---
Please modify the content in the script and config files, then run following command. 

`bash scripts/pretrain.sh`

### For Downstream Training 
---
We provide 2 different trainig strategies: Linear-probing, Full fine tuning. 
Please modify the content in the script and config files, then run following command. 

`bash scripts/downstream.sh`

### For Evaluation
---
Please modify the content in the script and config files, then run following command. 

`bash scripts/evaluation.sh`


## Main Experimental Results
---
We provide some experimental results on popular vision datasets. All experiments use IN-1k pre-trained I-JEPA model  (`in1k_vith14_ep300`) which is officially provided.  

| Dataset        | Linear Probing  | Full fine-tuning  |
|----------------|-----------------|-------------------|
| pets           | 89.2 [98.7]     | 83.6 [100]        |
| stanford_cars  | 25.6 [58.6]     | 10.94 [99.8]      |
| flowers102     | 82.4 [98.7]     | 90.80 [100.0]     |
| caltech101     | 93.5 [97.3]     | 92.60 [99.62]     |
| cub200         | 42.51 [85.10]   | 30.20 [99.97]     |

We report `test acc. [train acc.]` in each value in this table. 