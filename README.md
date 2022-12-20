# MEAformer
![](https://img.shields.io/badge/version-1.0.1-blue)
<!-- >[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)]() -->

 - [*MEAformer: Multi-modal Entity Alignment Transformer for Meta Modality Hybrid*]()

<!-- >In this paper .... -->

## Dependencies

- Python 3
- [PyTorch](http://pytorch.org/) (>= 1.6.0)
- [Transformers](http://huggingface.co/transformers/) (**version 3.0.2**)
- easydict
- unidecode
- tensorboard




## Train
- Using  `run.sh` file
```bash
>> cd MEAformer
>> bash run.sh
```
- Using  the bash command
```bash
>> cd MEAformer
# # w/ surface
# DBP15K
>> bash run_meaformer.sh 0 DBP15K zh_en 0.3 1 
>> bash run_meaformer.sh 0 DBP15K ja_en 0.3 1 
>> bash run_meaformer.sh 0 DBP15K fr_en 0.3 1 

# # w/o surface
# DBP15K
>> bash run_meaformer.sh 0 DBP15K zh_en 0.3 0 
>> bash run_meaformer.sh 0 DBP15K ja_en 0.3 0 
>> bash run_meaformer.sh 0 DBP15K fr_en 0.3 0
# FBYG15K
>> bash run_meaformer.sh 0 FBYG15K norm 0.8 0 
>> bash run_meaformer.sh 0 FBYG15K norm 0.5 0 
>> bash run_meaformer.sh 0 FBYG15K norm 0.2 0 
# FBDB15K
>> bash run_meaformer.sh 0 FBDB15K norm 0.8 0 
>> bash run_meaformer.sh 0 FBDB15K norm 0.5 0 
>> bash run_meaformer.sh 0 FBDB15K norm 0.2 0 
```

❗Tips: you can open the `run_meaformer.sh` file for parameter or training target modification.

## Dataset
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
├── model
│   ├── __init__.py
│   ├── layers.py
│   ├── MEAformer_loss.py
│   ├── MEAformer.py
│   ├── MEAformer_tools.py
│   └── Tool_model.py
├── requirement.txt
├── run_meaformer.sh
├── run.sh
├── src
│   ├── data_msnea.py
│   ├── data.py
│   ├── distributed_utils.py
│   ├── __init__.py
│   └── utils.py
├── torchlight
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
