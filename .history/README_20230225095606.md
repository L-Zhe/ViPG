# BTmPG
Code for paper [Visual Information Guided Zero-Shot Paraphrase Generation](https://aclanthology.org/2022.coling-1.568/) by Zhe Lin, Xiaojun Wan. This paper is accepted by Findings of COLING'22. Please contact me at [linzhe@pku.edu.cn](mailto:linzhe@pku.edu.cn) for any question.
<img src="https://github.com/L-Zhe/ViPG/blob/main/img/model.jpg?raw=true" width = "800" alt="overview" align=center />

## Dependencies
```
pytorch 1.4
StanfordCoreNLP
Pillow
numpy
Transformer
subword-nmt
```

## System Output

If you are looking for system output and don't bother to install dependencies and train a model (or run a pre-trained model), the [all-system-output](https://github.com/L-Zhe/ViPG/all-system-output) folder is for you.

## Train a new model
### Step 1: Preprocess

**Image**

You should first use ```gerate_img_feature.py``` to generate image feature vector. We note the output file of image feature vector as ```image_feature```

**Text**
You should first leverage ```POS.py``` to transform a text to the following format, and you will get the text file ```*.pos```.
<img src="https://github.com/L-Zhe/ViPG/blob/main/img/ner.jpg?raw=true" width = "800" alt="overview" align=center />


Then you can use the following command to apply byte pair encoding to word segmentation:

```
subword-nmt learn-bpe -s 32000 < *.pos > *.pos.code
subword-nmt apply-bpe -c *.pos.code < *.pos > *.pos.bpe
```

Next, you should generate vocabulary of corpora as follow:

```
python createVocab.py --file *.pos.bpe\
                      --lower --save_path ./vocab.share  --min_freq 1
```

Finally, you can employ the folloing command to generate the training data file:

```
python preprocess.py --sent_file *.pos.bpe \
                     --img_file image_file.img \
                     --vocab vocab.share \
                     --save_file train.data
```
```image_file.img``` is a text file, each line of it is an image file name which orresponds to the training file ```.pos.bpe```.

There are two examples of preprocessing dataset in ```./data/flickr``` and ```./data/mscoco```
### Step 2: Train Model

```
python train.py --cuda_num 2\
                --share_embed \
                --vocab ./vocab.share \
                --file ./train.data\
                --img_path ./image_feature/ \
                --checkpoint_path ./model \
                --checkpoint_n_epoch 1 \
                --grad_accum 1 \
                --max_tokens 5000 \
                --max_batch_size 256 \
                --discard_invalid_data
```

## Generation Paraphrase

```
python generator.py --cuda_num 1 \
                 --raw_file *.pos.bpe \
                 --ref_file *.test # we calculate self-bleu, so reference file is itself \
                 --max_tokens 300 \
                 --vocab ./vocab.share \
                 --decode_method beam \
                 --beam 10 \
                 --model_path ./model/checkpoint.pkl \
                 --output_path ./output \
                 --max_add_token 50 \
                 --max_alpha 1
```


## Pre-trained Models

We release our pretrained model at [here](https://github.com/L-Zhe/ViPG/releases/tag/model).


## Results

<img src="https://github.com/L-Zhe/ViPG/blob/main/img/result.jpg?raw=true" width = "800" alt="result" align=center />


## Citation

If you use any content of this repo for your work, please cite the following bib entry:

```
@inproceedings{lin-wan-2022-visual,
    title = "Visual Information Guided Zero-Shot Paraphrase Generation",
    author = "Lin, Zhe  and
      Wan, Xiaojun",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.568",
    pages = "6530--6539",
    abstract = "Zero-shot paraphrase generation has drawn much attention as the large-scale high-quality paraphrase corpus is limited. Back-translation, also known as the pivot-based method, is typical to this end. Several works leverage different information as {''}pivot{''} such as language, semantic representation and so on. In this paper, we explore using visual information such as image as the {''}pivot{''} of back-translation. Different with the pipeline back-translation method, we propose visual information guided zero-shot paraphrase generation (ViPG) based only on paired image-caption data. It jointly trains an image captioning model and a paraphrasing model and leverage the image captioning model to guide the training of the paraphrasing model. Both automatic evaluation and human evaluation show our model can generate paraphrase with good relevancy, fluency and diversity, and image is a promising kind of pivot for zero-shot paraphrase generation.",
}
```
