# ADRec

This is the official PyTorch implementation for the paper:

***Bidirectional Alignment Text-embeddings with Decoupled Contrastive for Sequential Recommendation***



## Overview
The key to sequential recommendation is to accurately predict the next item based on historical interaction sequences by learning sequence representations. Existing models optimize sequence representations directly using the next ground truth item as the supervised signal. However, they tend to focus on unidirectional semantic alignment with the specific next item. This focus results in biased interest representations and neglects the benefits of bidirectional supervision, leading to incomplete sequence representations and semantic mismatches. To address this problem, we propose a novel approach named ADRec for bidirectional sequence-item Alignment text-embeddings with Decoupled contrastive learning for sequential Recommendation based only on text data. The core idea is to utilize intrinsic correlations in recommendation data to derive self-supervised and supervised signals, enhancing the semantic consistency between sequence and target item representations and ultimately improving sequential recommendation. Specifically, we introduce a hybrid learning mechanism with an unsupervised contrastive learning paradigm to learn decoupled representations of sequences and items. Additionally, it incorporates supervised contrastive learning to learn bidirectional semantic alignment between sequences and items using interaction sequences and target items. Furthermore, we devise a dual-momentum queue mechanism to expand the range of negative samples obtainable with limited resources, optimizing the quality of user interest representations in the text modality. Extensive experiments on six public datasets demonstrate that ADRec achieves state-of-the-art performance over existing baselines by learning better sequence representations




## Requirements

python>=3.9.7

cudatoolkit>=11.3.1

pytorch>=1.11.0



## Quick Start



