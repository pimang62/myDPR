import os
import logging
import torch
from torch import tensor as T
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers import BertModel

transformers.logging.set_verbosity_error()  # 토크나이저 초기화 관련 warning suppress
from tqdm import tqdm
from typing import Tuple, List

from chunker import DataChunk, ArticleChunk
from encoder import KobertBiEncoder
import indexers
from utils import get_passage_file
import json
import pickle
import pandas as pd

# logger basic config
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/log.log",
    level=logging.DEBUG,
    format="[%(asctime)s | %(funcName)s @ %(pathname)s] %(message)s",
)
logger = logging.getLogger()


def passage_collator(batch: List, padding_value: int) -> Tuple[torch.Tensor]:
    """passage를 batch로 반환합니다."""
    batch_p = pad_sequence(
        [T(e) for e in batch], batch_first=True, padding_value=padding_value
    )
    batch_p_attn_mask = (batch_p != padding_value).long()
    return (batch_p, batch_p_attn_mask)


class ArticleStream(torch.utils.data.IterableDataset):
    """
    Indexing을 위해 random access가 필요하지 않고 large corpus를 다루기 위해 stream dataset을 사용합니다.
    """
    def __init__(self, article_path, chunker):
        super(ArticleStream).__init__()
        self.article_path = article_path
        self.chunker = chunker
        self.pad_token_id = self.chunker.tokenizer.get_vocab()["[PAD]"]
        self.max_length = 100  # maximum length

    def __iter__(self):
        # max_length가 되도록 padding 수행
        if self.article_path.split(".")[-1] == 'tsv':
            article = pd.read_csv(self.article_path, sep='\t')
            orig_texts, chunk_lists = [], []
            for _, item in article.iterrows():
                title_text = (item["title"].strip(), item["content"].strip() if item["content"] else item["answer"].strip())
                orig_text, chunk_list = self.chunker.chunk(title_text)
                # [[""]], [[]] -> [""], []
                orig_texts.append(orig_text[0])
                chunk_lists.append(chunk_list[0])
        else:
            orig_texts, chunk_lists = self.chunker.chunk(self.article_path)
        
        os.makedirs("article", exist_ok=True)
        index_name = self.article_path.split('.')[-2].split('/')[-1]
        with open(f"article/{index_name}.p", 'wb') as f:
            pickle.dump(orig_texts, f)
        
        logger.debug(f"chunked file {self.article_path}")
        for chunk in chunk_lists:
            yield chunk
    

class IndexRunner:
    """코퍼스에 대한 인덱싱을 수행하는 메인클래스입니다. passage encoder와 data loader, FAISS indexer로 구성되어 있습니다."""
    def __init__(
        self,
        article_path: str,
        model_ckpt_path: str,
        indexer_type: str = "DenseFlatIndexer",
        chunk_size: int = 100,
        chunk_overlap: int = 0,
        passages_type: str = "",
        batch_size: int = 64,
        buffer_size: int = 50000,
        index_output: str = "",
        device: str = "cuda:2" if torch.cuda.is_available() else "cpu",
    ):
        """
        article_path : 인덱싱할 article이 들어있는 경로입니다. 후에 article_dir로 변경할 예정입니다.
        indexer_type : 사용할 FAISS indexer 종류로 DPR 리포에 있는 대로 Flat, HNSWFlat, HNSWSQ 세 종류 중에 사용할 수 있습니다.
        chunk_size : indexing과 searching의 단위가 되는 passage의 길이입니다. DPR 논문에서는 100개 token 길이 (+ title)로 하나의 passage를 정의하였습니다.
        """
        self.device = torch.device(device)
        self.encoder = KobertBiEncoder().to(self.device)
        self.encoder.load(model_ckpt_path)  # loading model
        self.indexer = getattr(indexers, indexer_type)()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.article_path = article_path
        # article_path에서 떼다가 naming
        self.index_name = article_path.split('.')[-2].split('/')[-1]  
        self.loader = self.get_loader(
            self.article_path,
            chunk_size,
            chunk_overlap,
            batch_size,
            worker_init_fn=None,
        )
        self.indexer.init_index(self.encoder.emb_sz)
        self.index_output = index_output if index_output else indexer_type

    @staticmethod  # C().f() -> C.f() available
    def get_loader(article_path, chunk_size, chunk_overlap, batch_size, worker_init_fn=None):
        if article_path.split(".")[-1] == 'tsv':
            chunker = DataChunk(chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap)
        else:  # txt, pdf, docx
            chunker = ArticleChunk(chunk_size=chunk_size,
                                   chunk_overlap=chunk_overlap)
        ds = torch.utils.data.ChainDataset(
            tuple(ArticleStream(article_path, chunker) for article_path in [article_path])  # 나중에 data_dir로 교체
        )
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            collate_fn=lambda x: passage_collator(
                x, padding_value=chunker.tokenizer.get_vocab()["[PAD]"]
            ),
            num_workers=1,
            worker_init_fn=worker_init_fn,
        )
        return loader


    def run(self):
        _to_index = []
        cur = 0
        for batch in tqdm(self.loader, desc="indexing"):
            p, p_mask = batch
            p, p_mask = p.to(self.device), p_mask.to(self.device)
            with torch.no_grad():
                p_emb = self.encoder(p, p_mask, "passage")
            _to_index += [(cur + i, _emb) for i, _emb in enumerate(p_emb.cpu().numpy())]
            cur += p_emb.size(0)
            if len(_to_index) > self.buffer_size - self.batch_size:
                logger.debug(f"perform indexing... {len(_to_index)} added")
                self.indexer.index_data(_to_index)
                _to_index = []
        if _to_index:
            logger.debug(f"perform indexing... {len(_to_index)} added")
            self.indexer.index_data(_to_index)
            _to_index = []
        os.makedirs(self.index_output, exist_ok=True)
        self.indexer.serialize(self.index_output, self.index_name)


if __name__ == "__main__":
    """
    python index_runner.py --article_path "dataset/baemin_qc.tsv" \
                           --model_name "model/my_model.pt"
    """
    import argparse
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--article_path", type=str, default="dataset/baemin_qc.tsv", required=True)
    parser.add_argument("--model_name", type=str, default="model/my_model.pt", required=True)

    args = parser.parse_args()

    IndexRunner(
        article_path=args.article_path,
        model_ckpt_path=args.model_name,
        index_output="index",
        chunk_size=100,
        chunk_overlap=0
    ).run()
