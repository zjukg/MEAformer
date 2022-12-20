# MEAformer
![](https://img.shields.io/badge/version-1.0.1-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/zjukg/DUET/blob/main/licence)
 - [*MEAformer: Multi-modal Entity Alignment Transformer for Meta Modality Hybrid*]()

<!-- >In this paper .... -->

### Overall Path
❗NOTE: Please organize your files hierarchically as follows:
```
ROOT
├── data
│   └── mmkg
├── code
│   └── MEAformer
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
