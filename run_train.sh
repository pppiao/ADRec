######## 下面是DE相关的语言模型
# pretrain_data_process/Instruments_process/Scientifics_process/Games_process/Arts_process/Offices_process/Pets_process
# checkpoint_Games_16_1000
python main.py  --device 0 \
                --batchSize 64 \
                --dataset Scientifics_process \
                --save_dir_root checkpoint_Scientifics_64_3layer_4e4_feedfoward1024_动量队列_10000_bert12\
                --lr 0.0004 \
                --lr_dc_step 4 \
                --epoch 3 \
                --queueSize 10000 \
                --text_length 60 \
                --plm_hiddenSize 768 \
                --hiddenSize 768 \
                # --is_queue_warm_up false

