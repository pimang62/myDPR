#!/bin/bash

nohup python trainer.py --model_name "model/kobert_biencoder.pt" \
                        --train_path "dataset/train_baemin_qc.tsv" \
                        --valid_path "dataset/valid_baemin_qc.tsv" \
                        --passages_dir "passages" \
                        --outputs_dir "model/my_model.pt" \
                        --title_index_map_path "title_passage_map.p" \
                        --bm25 False > trial.log
