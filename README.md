# ADRec

This is the official PyTorch implementation for the paper:

***Bidirectional Alignment Text-embeddings with Decoupled Contrastive for Sequential Recommendation***



## Overview
The key to sequential recommendation is to accurately predict the next item based on historical interaction sequences by learning sequence representations. Existing models optimize sequence representations directly using the next ground truth item as the supervised signal. However, they tend to focus on unidirectional semantic alignment with the specific next item. This focus results in biased interest representations and neglects the benefits of bidirectional supervision, leading to incomplete sequence representations and semantic mismatches. To address this problem, we propose a novel approach named ADRec for bidirectional sequence-item Alignment text-embeddings with Decoupled contrastive learning for sequential Recommendation based only on text data. The core idea is to utilize intrinsic correlations in recommendation data to derive self-supervised and supervised signals, enhancing the semantic consistency between sequence and target item representations and ultimately improving sequential recommendation. Specifically, we introduce a hybrid learning mechanism with an unsupervised contrastive learning paradigm to learn decoupled representations of sequences and items. Additionally, it incorporates supervised contrastive learning to learn bidirectional semantic alignment between sequences and items using interaction sequences and target items. Furthermore, we devise a dual-momentum queue mechanism to expand the range of negative samples obtainable with limited resources, optimizing the quality of user interest representations in the text modality. Extensive experiments on six public datasets demonstrate that ADRec achieves state-of-the-art performance over existing baselines by learning better sequence representations.
![image](https://github.com/user-attachments/assets/2c94ce44-30b0-4b4b-ae90-b23bd9557fc8)


## Requirements

python>=3.9.7

cudatoolkit>=11.3.1

pytorch>=1.11.0

## Quick Start
### **Download Datasets and Model Outputs**  
1. Download the processed datasets and the model outputs generated during training from the following Google Drive link:  
   [Processed Datasets and Model Outputs]([https://drive.google.com/file/d/1ckwcigDvkQ7lvOJIIpNdwvD-tHt17WIS/view?usp=drive_link](https://drive.google.com/drive/folders/14rvjCYZ0EBa8AkKvQ1_9y0CCBbE-em-_?dmr=1&ec=wgc-drive-hero-goto))  

2. After downloading, extract the files and save them into directories with names matching the corresponding compressed file names. The expected directory structure is as follows:  
   - `datasets/`  
   - `embedding_out/`  
   - `transformer_with_pretrain/`  
   - `checkpoint_Arts_64_3layer_4e4_feedfoward1024_momentum_queue_10000_bert1/`  

   Ensure that the folder names match exactly to avoid errors during execution.

### **Training and Validation**
To start the training process, run the following command:  
```bash
bash run_train.sh

bash run_test.sh


