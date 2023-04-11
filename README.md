# MEAformer
![](https://img.shields.io/badge/version-1.0.1-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/zjukg/MEAformer/blob/main/licence)
[![arxiv badge](https://img.shields.io/badge/arxiv-2212.14454-red)](https://arxiv.org/abs/2212.14454)
[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)

 - [*MEAformer: Multi-modal Entity Alignment Transformer for Meta Modality Hybrid*](https://arxiv.org/abs/2212.14454)

<!-- >In this paper .... -->

## 🔬 Dependencies

- Python (>= 3.7)
- [PyTorch](http://pytorch.org/) (>= 1.6.0)
- numpy (>= 1.19.2)
- [Transformers](http://huggingface.co/transformers/) (>= 4.21.3)
- easydict (>= 1.10)
- unidecode (>= 1.3.6)
- tensorboard (>= 2.11.0)




## 🚀 Train
- Using  script file (`run.sh`)
```bash
>> cd MEAformer
>> bash run.sh
```
- Using the `bash command`
```bash
>> cd MEAformer
# -----------------------
# ---- non-iterative ----
# -----------------------
# ----  w/o surface  ---- 
# FBDB15K
>> bash run_meaformer.sh 1 FBDB15K norm 0.8 0 
>> bash run_meaformer.sh 1 FBDB15K norm 0.5 0 
>> bash run_meaformer.sh 1 FBDB15K norm 0.2 0 
# FBYG15K
>> bash run_meaformer.sh 1 FBYG15K norm 0.8 0 
>> bash run_meaformer.sh 1 FBYG15K norm 0.5 0 
>> bash run_meaformer.sh 1 FBYG15K norm 0.2 0 
# DBP15K
>> bash run_meaformer.sh 1 DBP15K zh_en 0.3 0 
>> bash run_meaformer.sh 1 DBP15K ja_en 0.3 0 
>> bash run_meaformer.sh 1 DBP15K fr_en 0.3 0
# ----  w/ surface  ---- 
# DBP15K
>> bash run_meaformer.sh 1 DBP15K zh_en 0.3 1 
>> bash run_meaformer.sh 1 DBP15K ja_en 0.3 1 
>> bash run_meaformer.sh 1 DBP15K fr_en 0.3 1
# -----------------------
# ------ iterative ------
# -----------------------
# ----  w/o surface  ---- 
# FBDB15K
>> bash run_meaformer_il.sh 1 FBDB15K norm 0.8 0 
>> bash run_meaformer_il.sh 1 FBDB15K norm 0.5 0 
>> bash run_meaformer_il.sh 1 FBDB15K norm 0.2 0 
# FBYG15K
>> bash run_meaformer_il.sh 1 FBYG15K norm 0.8 0 
>> bash run_meaformer_il.sh 1 FBYG15K norm 0.5 0 
>> bash run_meaformer_il.sh 1 FBYG15K norm 0.2 0 
# DBP15K
>> bash run_meaformer_il.sh 1 DBP15K zh_en 0.3 0 
>> bash run_meaformer_il.sh 1 DBP15K ja_en 0.3 0 
>> bash run_meaformer_il.sh 1 DBP15K fr_en 0.3 0
# ----  w/ surface  ---- 
# DBP15K
>> bash run_meaformer_il.sh 1 DBP15K zh_en 0.3 1 
>> bash run_meaformer_il.sh 1 DBP15K ja_en 0.3 1 
>> bash run_meaformer_il.sh 1 DBP15K fr_en 0.3 1
```

❗Tips: you can open the `run_meaformer.sh` or `run_meaformer_il.sh` file for parameter or training target modification.

## 📚 Dataset
❗NOTE: Download from [GoogleDrive](https://drive.google.com/file/d/1VIWcc3KDcLcRImeSrF2AyhetBLq_gsnx/view?usp=sharing) (1.26G) and unzip it to make those files satisfy the following file hierarchically:
```
ROOT
├── data
│   └── mmkg
└── code
    └── MEAformer
```

#### Code Path
```
MEAformer
├── config.py
├── main.py
├── requirement.txt
├── run_meaformer.sh
├── run_meaformer_il.sh
├── run.sh
├── model
│   ├── __init__.py
│   ├── layers.py
│   ├── MEAformer_loss.py
│   ├── MEAformer.py
│   ├── MEAformer_tools.py
│   └── Tool_model.py
├── src
│   ├── __init__.py
│   ├── distributed_utils.py
│   ├── data.py
│   └── utils.py
└── torchlight
    ├── __init__.py
    ├── logger.py
    ├── metric.py
    └── utils.py
```




#### Data Path
```
mmkg
├── DBP15K
│   ├── fr_en
│   │   ├── ent_ids_1
│   │   ├── ent_ids_2
│   │   ├── ill_ent_ids
│   │   ├── training_attrs_1
│   │   ├── training_attrs_2
│   │   ├── triples_1
│   │   └── triples_2
│   ├── ja_en
│   │   ├── ent_ids_1
│   │   ├── ent_ids_2
│   │   ├── ill_ent_ids
│   │   ├── training_attrs_1
│   │   ├── training_attrs_2
│   │   ├── triples_1
│   │   └── triples_2
│   ├── translated_ent_name
│   │   ├── dbp_fr_en.json
│   │   ├── dbp_ja_en.json
│   │   └── dbp_zh_en.json
│   └── zh_en
│       ├── ent_ids_1
│       ├── ent_ids_2
│       ├── ill_ent_ids
│       ├── training_attrs_1
│       ├── training_attrs_2
│       ├── triples_1
│       └── triples_2
├── FBDB15K
│   └── norm
│       ├── ent_ids_1
│       ├── ent_ids_2
│       ├── ill_ent_ids
│       ├── training_attrs_1
│       ├── training_attrs_2
│       ├── triples_1
│       └── triples_2
├── FBYG15K
│   └── norm
│       ├── ent_ids_1
│       ├── ent_ids_2
│       ├── ill_ent_ids
│       ├── training_attrs_1
│       ├── training_attrs_2
│       ├── triples_1
│       └── triples_2
├── embedding
│   └── glove.6B.300d.txt
├── pkls
│   ├── dbpedia_wikidata_15k_dense_GA_id_img_feature_dict.pkl
│   ├── dbpedia_wikidata_15k_norm_GA_id_img_feature_dict.pkl
│   ├── FBDB15K_id_img_feature_dict.pkl
│   ├── FBYG15K_id_img_feature_dict.pkl
│   ├── fr_en_GA_id_img_feature_dict.pkl
│   ├── ja_en_GA_id_img_feature_dict.pkl
│   └── zh_en_GA_id_img_feature_dict.pkl
├── MEAformer
└── dump
```

## Cite:
Please condiser citing this paper if you use the ```code``` or ```data``` from our work.
Thanks a lot :)

```bigquery
@article{DBLP:journals/corr/abs-2212-14454,
  author    = {Zhuo Chen and
               Jiaoyan Chen and
               Wen Zhang and
               Lingbing Guo and
               Yin Fang and
               Yufeng Huang and
               Yuxia Geng and
               Jeff Z. Pan and
               Wenting Song and
               Huajun Chen},
  title     = {MEAformer: Multi-modal Entity Alignment Transformer for Meta Modality
               Hybrid},
  journal   = {CoRR},
  volume    = {abs/2212.14454},
  year      = {2022}
}
```
