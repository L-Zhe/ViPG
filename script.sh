python createVocab.py --file /home/linzhe/data/big_with_google/cap_big.pos.bpe\
                      --lower --save_path ./data_google/vocab.share  --min_freq 1

python preprocess.py --sent_file /home/linzhe/data/big_with_google/cap_big.pos.bpe \
                     --img_file /home/linzhe/data/big_with_google/img_big.img \
                     --vocab ./data_google/vocab.share \
                     --save_file ./data_google/train.data

python preprocess.py --sent_file /home/linzhe/data/mscoco/process_data/captions_val2017.cap.pos.bpe  \
                     --vocab ./data_new/vocab.share \
                     --save_file ./data_new/mscoco.test


python train.py --cuda_num 3\
                --share_embed \
                --vocab ./data_google/vocab.share \
                --file ./data_google/train.data\
                --img_path /home/linzhe/DownloadConceptualCaptions-master/train_image_feature/ \
                --checkpoint_path ./data_google/model \
                --checkpoint_n_epoch 1 \
                --grad_accum 1 \
                --max_tokens 5000 \
                --max_batch_size 256 \
                --discard_invalid_data \
                --restore_file ./data_new/model/checkpoint.pkl

python train.py --cuda_num 2\
                --share_embed \
                --vocab ./data_mscoco/vocab.share \
                --file ./data_mscoco/train.data\
                --img_path /home/linzhe/DownloadConceptualCaptions-master/train_image_feature/ \
                --checkpoint_path ./data_tmp/model \
                --checkpoint_n_epoch 1 \
                --grad_accum 1 \
                --max_tokens 5000 \
                --max_batch_size 256 \
                --discard_invalid_data \
                --restore_file ./data_new/model/checkpoint.pkl

python train.py --cuda_num 3\
                --share_embed \
                --vocab ./data_flickr/vocab.share \
                --file ./data_flickr/train.data\
                --img_path /home/linzhe/DownloadConceptualCaptions-master/train_image_feature/ \
                --checkpoint_path ./data_flickr/model \
                --checkpoint_n_epoch 1 \
                --grad_accum 1 \
                --max_tokens 5000 \
                --max_batch_size 256 \
                --discard_invalid_data \
                --restore_file ./data_new/model/checkpoint.pkl

python generator.py --cuda_num 1 \
                 --raw_file /home/linzhe/data/flickr30k/flickr.val.pos.all.bpe \
                 --ref_file /home/linzhe/data/flickr30k/flickr.val \
                 --max_tokens 1000 \
                 --vocab ./data_google/vocab.share \
                 --decode_method beam \
                 --beam 10 \
                 --model_path ./data_google/model/checkpoint58.pkl \
                 --output_path ./data_google/output \
                 --max_add_token 50 \
                 --max_alpha 1

python generator.py --cuda_num 1 \
                 --raw_file /home/linzhe/data/big_with_google/flickr.pos.bpe \
                 --ref_file /home/linzhe/data/big_with_google/flickr.val \
                 --max_tokens 300 \
                 --vocab ./data_google/vocab.share \
                 --decode_method beam \
                 --beam 10 \
                 --model_path ./data_google/model/checkpoint58.pkl \
                 --output_path ./data_google/output \
                 --max_add_token 50 \
                 --max_alpha 1


python generator.py --cuda_num 3 \
                 --raw_file /home/linzhe/data/mscoco/process_data/captions_val2017.cap.pos.all.bpe \
                 --ref_file /home/linzhe/data/mscoco/process_data/captions_val2017.cap \
                 --max_tokens 1000 \
                 --vocab ./data_mscoco/vocab.share \
                 --decode_method beam \
                 --beam 5 \
                 --model_path ./data_mscoco/model/checkpoint.pkl \
                 --output_path ./data_mscoco/output \
                 --max_add_token 50 \
                 --max_alpha 1.5

python generator.py --cuda_num 1 \
                 --raw_file /home/linzhe/data/flickr30k/flickr.val.pos.all.bpe \
                 --ref_file /home/linzhe/data/flickr30k/flickr.val \
                 --max_tokens 300 \
                 --vocab ./data_flickr/vocab.share \
                 --decode_method beam \
                 --beam 15 \
                 --model_path ./data_flickr/model/checkpoint.pkl \
                 --output_path ./data_flickr/output_flickr \
                 --max_add_token 50 \
                 --max_alpha 1.5


python generator.py --cuda_num 3 \
                 --raw_file /home/linzhe/data/mscoco/process_data/captions_val2017.cap.pos.all.bpe \
                 --ref_file /home/linzhe/data/mscoco/process_data/captions_val2017.cap \
                 --max_tokens 1000 \
                 --vocab ./data_new/vocab.share \
                 --decode_method beam \
                 --beam 10 \
                 --model_path ./data_new/model/checkpoint.pkl \
                 --output_path ./data_new/output \
                 --max_add_token 50 \
                 --max_alpha 1.5

python generator.py --cuda_num 1 \
                 --raw_file /home/linzhe/data/flickr30k/flickr.val.pos.all.bpe \
                 --ref_file /home/linzhe/output/flickr_output/flickr.src \
                 --max_tokens 1000 \
                 --vocab ./data_new/vocab.share \
                 --decode_method beam \
                 --beam 5 \
                 --model_path ./data_new/model/checkpoint.pkl \
                 --output_path ./data_new/output \
                 --max_add_token 50 \
                 --max_alpha 1.5
# de2en

python avg_param.py --input ./data_google/model/checkpoint62.pkl \
                            ./data_google/model/checkpoint61.pkl \
                            ./data_google/model/checkpoint60.pkl \
                            ./data_google/model/checkpoint59.pkl \
                            ./data_google/model/checkpoint58.pkl \
                    --output ./data_google/model/checkpoint_emsemble.pkl

subword-nmt learn-bpe -s 32000 < /home/linzhe/data/big_with_google/cap_big.pos > /home/linzhe/data/big_with_google/cap_big.pos.code
subword-nmt apply-bpe -c /home/linzhe/data/big_with_google/cap_big.pos.code < /home/linzhe/data/flickr30k/flickr.val.pos > /home/linzhe/data/big_with_google/flickr.pos.bpe
subword-nmt apply-bpe -c /home/linzhe/data/flickr30k/flickr.all.pos.code < /home/linzhe/data/flickr30k/flickr.val.pos > /home/linzhe/data/flickr30k/flickr.val.pos.all.bpe
java -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit -file /home/linzhe/data/mscoco/process_data/captions_train2017.cap -outputFormat conll -output.columns word
python tools/eval.py --model model-best.pth --infos_path infos_trans_nscl-best.pkl --image_folder /home/linzhe/bt_image/T2I_CL/DM-GAN+CL/data/coco --num_images 10

bert-score -r /home/linzhe/output/mscoco_output/our_one_caption_google.mscoco -c /home/linzhe/output/mscoco_output/mscoco.src --lang en --rescale_with_baseline
bert-score -r /home/linzhe/output/flickr_output/our_one_caption_google.flickr  -c /home/linzhe/output/flickr_output/flickr.src --lang en --rescale_with_baseline



subword-nmt apply-bpe -c /home/linzhe/data/big_with_google/cap_big.pos.code < /home/linzhe/data/para-nmt-50m/para-nmt-8k.src.pos > //home/linzhe/data/para-nmt-50m/para-nmt-8k.src.pos.bpe
python generator.py --cuda_num 3 \
                 --raw_file /home/linzhe/data/para-nmt-50m/para-nmt-8k.src.pos.bpe \
                 --ref_file /home/linzhe/data/para-nmt-50m/para-nmt-8k.src \
                 --max_tokens 300 \
                 --vocab ./data_google/vocab.share \
                 --decode_method beam \
                 --beam 10 \
                 --model_path ./data_google/model/checkpoint58.pkl \
                 --output_path ./data_google/output \
                 --max_add_token 50 \
                 --max_alpha 1
