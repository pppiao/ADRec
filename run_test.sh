export check_epoch=2
export device=0
export save_dir_root="checkpoint_Instruments_64_3layer_4e4_feedfoward1024_动量队列_256"
export dataset="Instruments_process"

cd embedding_out

# Instruments_process/Scientifics_process/Games_process/Arts_process/Offices_process/Pets_process
python test_main.py --device ${device} \
                    --dataset ${dataset} \
                    --save_dir_root ${save_dir_root} \
                    --check_epoch ${check_epoch} \
                    --text_length 60 \
                    --plm_hiddenSize 768 \
                    --hiddenSize 768 
cd ..

cd transformer_with_pretrain

python test_main.py \
    --device ${device} \
    --save_dir ${save_dir_root} \
    --dataset ${dataset} \
    --check_epoch ${check_epoch} \
    --hiddenSize 768
cd ..


