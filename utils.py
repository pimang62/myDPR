from glob import glob
from typing import List
from torch import tensor as T
from rank_bm25 import BM25Okapi
import string
from tqdm import tqdm

def get_passage_file(p_id_list: List[int]) -> str:
    """passage id를 받아서 해당되는 파일 이름을 반환합니다."""
    target_file = None
    p_id_max = max(p_id_list)
    p_id_min = min(p_id_list)
    for f in glob(f"passages/*.p"):  # passages_dir/*-*.p
        s, e = f.split("/")[-1].split(".")[0].split("-")
        s, e = int(s), int(e)
        if p_id_min >= s and p_id_max <= e:
            target_file = f
    return target_file


def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)
        # if len(token) > 5:
        tokenized_doc.append(token)
    return tokenized_doc


    