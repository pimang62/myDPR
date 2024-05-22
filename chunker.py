import os
import logging
import re
import json
import pandas as pd
import pickle
from typing import Tuple, List
from kobert_tokenizer import KoBERTTokenizer
from glob import glob
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict
from chunk import ChunkerFactory

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/log.log",
    level=logging.DEBUG,
    format="[%(asctime)s | %(funcName)s @ %(pathname)s] %(message)s",
)
logger = logging.getLogger()


def post_processing(text:str) -> str:
    return re.sub(r'\.', '', text)


class DataChunk:
    """
    trainer용 input data 형태를 따르는 chunk 클래스입니다.
    >>> pd.DataFrame({"title":[str], "question":[str], "content" or "asnwer":[str]})
    
    Option
    1. chunk_size: 100, 200, 50 ...
    2. chunk_overlap: 0, 5, 10, ...
    """
    def __init__(self, chunk_size: int=100, chunk_overlap: int=0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")

    def chunk(self, title_text: Tuple[str, str]) -> Tuple[List[str], List[int]]:
        title, text = title_text
        """chunk_size, chunk_overlap 유효"""
        chunk_list = []
        orig_text = []

        encoded_title = self.tokenizer.encode(title)
        encoded_text = self.tokenizer.encode(text)

        for start_idx in range(0, len(encoded_text), self.chunk_size-self.chunk_overlap):
            end_idx = min(len(encoded_text), start_idx + self.chunk_size)
            chunk = encoded_title + encoded_text[start_idx:end_idx]
            orig_text.append(self.tokenizer.decode(chunk))
            chunk_list.append(chunk)
        
        return orig_text, chunk_list


def save_orig_passage(
    data_path,
    passages_dir, 
    chunk_size, 
    chunk_overlap,
):
    """store original passages with unique id"""
    os.makedirs(passages_dir, exist_ok=True)
    chunker = DataChunk(chunk_size=chunk_size, 
                        chunk_overlap=chunk_overlap)
    
    idx = 0
    data_frame = pd.read_csv(data_path, sep='\t')
    for _, sample in data_frame.iterrows():
        title_text = (sample["title"].strip(), sample["content"].strip() if "content" in data_frame.columns else sample["answer"].strip())  # "content" or "answer"
        orig, _ = chunker.chunk(deepcopy(title_text))
        to_save = {idx + i: orig[i] for i in range(len(orig))}
        with open(f"{passages_dir}/{idx}-{idx+len(orig)-1}.p", "wb") as f:
            pickle.dump(to_save, f)
        idx += len(orig)
    logger.info(f"Finished saving original passages at {passages_dir}!")


def save_title_index_map(
    title_index_map_path, 
    passages_dir
):
    """korquad와 klue 데이터 전처리를 위해 title과 passage id를 맵핑합니다.
    title_index_map : dict[str, list] 형태로, 특정 title에 해당하는 passage id를 저장합니다.
    """
    logging.getLogger()

    files = glob(f"{passages_dir}/*")
    title_id_map = defaultdict(list)
    for file in tqdm(files):
        with open(file, "rb") as _f:
            id_passage_map = pickle.load(_f)
        for id, passage in id_passage_map.items():
            title = passage.split("[SEP]")[0].split("[CLS]")[1].strip()
            title_id_map[title].append(id)
        logger.info(f"processed {len(id_passage_map)} passages from {_f}...")
    with open(f"{title_index_map_path}", "wb") as f:
        pickle.dump(title_id_map, f)
    logger.info(f"Finished saving title_index_mapping at {title_index_map_path}!")


class ArticleChunk:
    """
    index_runner용 Article을 자르는 chunk 클래스입니다.
    >>> article_path가 들어옵니다.
    """
    def __init__(self, chunk_size: int=None, chunk_overlap: int=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")

    def chunk(self, article_path):
        """input file format은 txt, pdf, docx 중의 하나의 형태를 따릅니다."""
        chunker = ChunkerFactory(article_path)
        texts = chunker.create_chunker().chunk(chunk_size=self.chunk_size,
                                                    chunk_overlap=self.chunk_overlap)
        
        chunk_list = []
        orig_text = []
        for text in texts:
            text = text.strip()
            if not text:
                logger.debug(f"article is empty, passing")
                continue

            encoded_text = self.tokenizer.encode(text)
            if len(encoded_text) < 5:  # 본문 길이가 subword 5개 미만인 경우 패스
                logger.debug(f"article: {text} has <5 subwords in its article, passing")
                continue

            # article마다 chunk_size 길이의 chunk를 만들어 list에 append. (각 chunk에는 title을 prepend합니다.)
            # ref : DPR paper
            if self.chunk_size is not None:
                for start_idx in range(0, len(encoded_text), self.chunk_size-self.chunk_overlap):
                    end_idx = min(len(encoded_text), start_idx + self.chunk_size)
                    chunk = encoded_text[start_idx:end_idx]
                    orig_text.append(self.tokenizer.decode(chunk))
                    chunk_list.append(chunk)
            else:
                orig_text.append(text)
                chunk_list.append(encoded_text)

        return orig_text, chunk_list


if __name__=='__main__':
    """"
    python chunker.py --data_path "dataset/baemin_qc.tsv" \
                      --passages_dir "passages" \
                      --title_index_map_path "title_passage_map.p" \
                      --chunk_size 100 \
                      --chunk_overlap 0
    """
    import argparse
    parser = argparse.ArgumentParser(description="Data Chunk")
    parser.add_argument("--data_path", type=str, default="dataset/baemin_qc.tsv", required=True)
    parser.add_argument("--passages_dir", type=str, default="passages", required=True)
    parser.add_argument("--title_index_map_path", type=str, default="title_passage_map.p", required=True)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--chunk_overlap", type=int, default=0)

    args = parser.parse_args()

    save_orig_passage(
        data_path=args.data_path,
        passages_dir=args.passages_dir, 
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap, 
    )

    save_title_index_map(
        title_index_map_path=args.title_index_map_path, 
        passages_dir=args.passages_dir
    )

