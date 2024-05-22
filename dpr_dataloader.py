from tqdm import tqdm
import torch
from torch import tensor as T
from torch.nn.utils.rnn import pad_sequence
import os
import json
import re
import logging
from typing import Iterator, List, Sized, Tuple
import pickle
from kobert_tokenizer import KoBERTTokenizer
import pandas as pd
from glob import glob

from utils import (
    get_passage_file,
    bm25_tokenizer
)
from rank_bm25 import BM25Okapi

# set logger
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/log.log",
    level=logging.DEBUG,
    format="[%(asctime)s | %(funcName)s @ %(pathname)s] %(message)s",
)
logger = logging.getLogger()


def data_collator(batch: List[Tuple], padding_value: int) -> Tuple[torch.Tensor]:
    """query, p_id, gold_passage를 batch로 반환합니다."""
    batch_q = pad_sequence(
        [T(e[0]) for e in batch], batch_first=True, padding_value=padding_value
    )
    batch_q_attn_mask = (batch_q != padding_value).long()
    batch_p_id = T([e[1] for e in batch])[:, None]
    batch_p = pad_sequence(
        [T(e[2]) for e in batch], batch_first=True, padding_value=padding_value
    )
    batch_p_attn_mask = (batch_p != padding_value).long()
    return (batch_q, batch_q_attn_mask, batch_p_id, batch_p, batch_p_attn_mask)


class OnehotSampler(torch.utils.data.BatchSampler):
    def __init__(
        self,
        data_source: Sized,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        generator=None,
    ) -> None:
        if shuffle:
            sampler = torch.utils.data.RandomSampler(
                data_source, replacement=False, generator=generator
            )
        else:
            sampler = torch.utils.data.SequentialSampler(data_source)
        super(OnehotSampler, self).__init__(
            sampler=sampler, batch_size=batch_size, drop_last=drop_last
        )
        
        def __iter__(self) -> Iterator[List[int]]:
            sampled_p_id = []
            sampled_idx = []
            for idx in self.sampler:
                item = self.sampler.data_source[idx]
                if item[1] in sampled_p_id:
                    continue  # 만일 같은 answer passage가 이미 뽑혔다면 pass
                sampled_idx.append(idx)
                sampled_p_id.append(item[1])
                if len(sampled_idx) >= self.batch_size:
                    yield sampled_idx  # batch로 넘기고 초기화
                    sampled_p_id = []
                    sampled_idx = []
            if len(sampled_idx) > 0 and not self.drop_last:
                yield sampled_idx  # 남은 batch 손실하지 않고 사용


class BM25Sampler(torch.utils.data.BatchSampler):
    def __init__(
        self,
        data_source: Sized,
        batch_size: int,
        drop_last: bool = False,
    ) -> None:
        # shuffle=False should be sequential
        sampler = torch.utils.data.SequentialSampler(data_source)
        super(BM25Sampler, self).__init__(
            sampler=sampler, batch_size=batch_size, drop_last=drop_last
        )
          
        def __iter__(self) -> Iterator[List[int]]:
            sampled_p_id = []
            sampled_idx = []
            for idx in self.sampler:
                item = self.sampler.data_source[idx]
                # if item[1] in sampled_p_id:
                #     continue  # 만일 같은 answer passage가 이미 뽑혔다면 pass
                sampled_idx.append(idx)
                sampled_p_id.append(item[1])
                if len(sampled_idx) >= self.batch_size:
                    yield sampled_idx  # batch로 넘기고 초기화
                    sampled_p_id = []
                    sampled_idx = []
            if len(sampled_idx) > 0 and not self.drop_last:
                yield sampled_idx  # 남은 batch 손실하지 않고 사용


class Dataset:
    # baemin_path : 'dataset/baemin.json'
    def __init__(self,
                 model_name: str="skt/kobert-base-v1",
                 data_path: str="dataset/baemin_qc.tsv", 
                 passages_dir: str="passages",
                 title_index_map_path: str="title_passage_map.p",
                 bm25: bool = True):
        self.data_path = data_path
        self.passages_dir = passages_dir
        self.title_index_map_path = title_index_map_path
        self.data_tuples = []
        self.tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
        self.pad_token_id = self.tokenizer.get_vocab()["[PAD]"]
        self.bm25 = bm25
        self.load()

    @property
    def dataset(self) -> List[Tuple]:
        return self.tokenized_tuples

    def stat(self):
        """korquad 데이터셋의 스탯을 출력합니다."""
        raise NotImplementedError()

    def load(self):
        """데이터 전처리가 완료되었다면 load하고 그렇지 않으면 전처리를 수행합니다."""
        if self.bm25:
            self.data_processed_path = (
                f"dataset/{self.data_path.split('.')[0].split('/')[-1]}_processed_bm25.p"
            )
        else:
            self.data_processed_path = (
                f"dataset/{self.data_path.split('.')[0].split('/')[-1]}_processed.p"
            )
        if os.path.exists(f"dataset/{self.data_processed_path}"):
            logger.debug("preprocessed file already exists, loading...")
            with open(f"dataset/{self.data_processed_path}", "rb") as f:
                self.tokenized_tuples = pickle.load(f)
            logger.debug(
                "successfully loaded tokenized_tuples into self.tokenized_tuples"
            )

        else:
            self._load_data()
            self._match_passage()
            logger.debug("successfully loaded data_tuples into self.data_tuples")
            # tokenizing raw dataset
            self.tokenized_tuples = [
                (self.tokenizer.encode(q), id, self.tokenizer.encode(p))
                for q, id, p in tqdm(self.data_tuples, desc="tokenize")
            ]
            self._save_processed_dataset()
            logger.debug("finished tokenization")

    def _load_data(self):
        # if os.path.isfile(self.data_path):
        with open(self.data_path, "rt", encoding="utf8") as f:
            self.raw_data = pd.read_csv(self.data_path, sep='\t')
        # else:  # data_path가 곧바로 객체로 들어올 경우 => KFold 구별을 위해
        #     self.raw_data = self.data_path
        logger.debug(f"data loaded into self.raw_data")
        with open(self.title_index_map_path, "rb") as f:
            self.title_passage_map = pickle.load(f)
        logger.debug("title passage mapping loaded into self.title_passage_map")

    def _get_cand_ids(self, title):
        """baemin 데이터에서 해당 title에 맞는 id들을 가지고 옵니다."""
        ret = self.title_passage_map.get(title, None)  # List[int]
        return ret

    def _match_passage(self):
        """quetion과 answer를 매칭하여 (query, passage_id, passage)의 tuple을 구성합니다.
        >>> pd.DataFrame({"title":[str], "question":[str], "content" or "asnwer":[str]})"""

        all_passages = []  # 전체 "context" or "answer" 다 담기 => 4/29 해결해야 함 
        tokenized_corpus = []
        if not all_passages or not tokenized_corpus:
            logger.debug("passages directory exists, tokenizing...")
            for page in sorted(glob(f"passages/*.p")):
                with open(page, 'rb') as f:
                    page_map = pickle.load(f)
                all_passages.extend([pm for pm in page_map.values()])
            for example in all_passages:
                tokenized_corpus.append(bm25_tokenizer(example))

        bm25_selector = BM25Okapi(tokenized_corpus)

        for _, item in tqdm(self.raw_data.iterrows(), desc="matching silver passages"):
            title = item["title"].strip()  # "content" or "answer"
            cand_ids = self._get_cand_ids(title)
            if cand_ids is None:
                logger.debug(
                    f"No such title as {title} or {title}. passing this title"
                )
            target_file_p = get_passage_file(cand_ids)
            if target_file_p is None:
                logger.debug(
                    f"No single target file for {title}, got passage ids {cand_ids}. passing this title"
                )
                continue
            with open(target_file_p, "rb") as f:
                target_file = pickle.load(f)
            # passages_type이 content면 contexts가 하나! => contexts가 의미가 있나? (추후 생각)
            contexts = {cand_id: target_file[cand_id] for cand_id in cand_ids}

            query = item["question"].strip()
            passage = item["content"].strip() if "content" in self.raw_data.columns else item["answer"].strip() # "content" or "answer"
            
            # tokenized answer
            encoded_title = self.tokenizer.encode(title)
            encoded_answer = self.tokenizer.encode(passage)
            decoded_answer = self.tokenizer.decode(encoded_title + encoded_answer)[:10]

            if self.bm25:  # bm25 적용
                bm25_scores = bm25_selector.get_scores(bm25_tokenizer(decoded_answer))

                all_sentence_combinations = []
                for i in range(len(bm25_scores)):
                    all_sentence_combinations.append([bm25_scores[i], i])

                # sort the score descending
                all_sentence_combinations = sorted(all_sentence_combinations, key=lambda x: -x[0])  # 내림차순!

                negative_answer = ''
                for j in range(len(all_sentence_combinations)):
                    jdx = all_sentence_combinations[j][1]
                    if decoded_answer not in all_passages[jdx]:  # token 잘릴 수 있어서
                        negative_answer = all_passages[jdx]
                        break  # 중복 처리
                
                import random
                answer_p = [
                    (p_id, c) for p_id, c in contexts.items() if decoded_answer in c
                ]
                answer_p = [random.choice(answer_p)]  # answer가 단순히 들어있는 문서를 뽑는다.

                hard_negative_answer_p = [
                    (p_id, c) for p_id, c in contexts.items() if negative_answer in c
                ]
                
                if not hard_negative_answer_p:  # filterinig
                    logger.debug(
                        f"No hard_negative_answer in candidates, try to get all passages..."
                    )
                    hard_negative_answer_p = [
                        (p_id, c) for p_id, c in enumerate(all_passages) if negative_answer in c
                    ]

                hard_negative_answer_p = [random.choice(hard_negative_answer_p)]  # in hard negative answers

                self.data_tuples.extend(
                    [(query, p_id, c) for p_id, c in answer_p]
                )
                
                self.data_tuples.extend(
                    [(query, p_id, c) for p_id, c in hard_negative_answer_p]
                )
                
            else:  # not bm25
                answer_p = [
                    (p_id, c) for p_id, c in contexts.items() if decoded_answer in c
                ]
                import random
                answer_p = [random.choice(answer_p)]  # answer가 단순히 들어있는 문서를 뽑는다.

                self.data_tuples.extend(
                    [(query, p_id, c) for p_id, c in answer_p]
                )


    def _save_processed_dataset(self):
        """전처리한 데이터를 저장합니다."""
        with open(self.data_processed_path, "wb") as f:
            pickle.dump(self.tokenized_tuples, f)
        logger.debug(
            f"successfully saved self.tokenized_tuples into {self.data_processed_path}"
        )


if __name__ == "__main__":
    orig_ds = Dataset(# model_name="skt/kobert-base-v1",
                      data_path="dataset/baemin_qc.tsv", 
                      passages_dir="passages",
                      title_index_map_path="title_passage_map.p",
                      bm25=False)  # Random
    print(len(orig_ds.dataset))  # 987
    
    ds = Dataset(# model_name="skt/kobert-base-v1",
                 data_path="dataset/baemin_qc.tsv", 
                 passages_dir="passages",
                 title_index_map_path="title_passage_map.p",
                 bm25=True)  # BM25
    print(len(ds.dataset))  # 1974 : "2배여야

