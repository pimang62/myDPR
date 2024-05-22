import torch
from torch import tensor as T
import pickle
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from kobert_tokenizer import KoBERTTokenizer
from indexers import DenseFlatIndexer
from encoder import KobertBiEncoder
from dpr_dataloader import Dataset, OnehotSampler, data_collator
from utils import get_passage_file
from typing import List


class DPRRetriever:
    def __init__(self, model, indexer, article_path, val_batch_size: int = 64, device='cuda:2' if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        # self.tokenizer = valid_dataset.tokenizer
        self.tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
        self.val_batch_size = val_batch_size
        # self.valid_loader = torch.utils.data.DataLoader(
        #     dataset=valid_dataset.dataset,
        #     batch_sampler=OnehotSampler(
        #         valid_dataset.dataset, batch_size=val_batch_size, drop_last=False
        #     ),
        #     collate_fn=lambda x: data_collator(
        #         x, padding_value=valid_dataset.pad_token_id
        #     ),
        #     num_workers=4,
        # )
        self.indexer = indexer
        self.article_path = article_path
        # self.mode = mode


    def retrieve(self, query: str, k: int = 100):
        """주어진 쿼리에 대해 가장 유사도가 높은 passage를 반환합니다."""
        self.model.eval()  # 평가 모드
        tok = self.tokenizer.batch_encode_plus([query])
        with torch.no_grad():
            out = self.model(T(tok["input_ids"]).to(self.device), T(tok["attention_mask"]).to(self.device), "query")  # all tensors to be on the same device : .to(self.device)
        result = self.indexer.search_knn(query_vectors=out.cpu().numpy(), top_docs=k)
        
        # if self.mode == "train":
        # # 원문 가져오기 -> dataset이 아닌 txt chunk로 하고 싶음..
        #     passages = []
        #     for idx, sim in zip(*result[0]):
        #         path = get_passage_file([idx], passages_dir=valid_dataset.passages_dir)
        #         if not path:
        #             print(f"No single passage path for {idx}")
        #             continue
        #         with open(path, "rb") as f:
        #             passage_dict = pickle.load(f)
        #         print(f"문서 #{idx+1}, 점수: {sim}\n{passage_list[idx]}", end='\n\n')
        #         passages.append((passage_dict[idx], sim))
        #     return passages
        
        # else:  # self.mode == "article"
        passages = []
        for idx, sim in zip(*result[0]):
            with open(self.article_path, 'rb') as f:
                passage_list = pickle.load(f)
            print(f"문서 #{idx+1}, 점수: {sim}\n{passage_list[idx]}", end='\n\n')
            passages.append((passage_list[idx], sim))
        return passages


    # def val_top_k_acc(self, k:List[int]=[5] + list(range(10,101,10))):
    #     '''validation set에서 top k 정확도를 계산합니다.'''
    #     self.model.eval()  # 평가 모드
    #     k_max = max(k)
    #     sample_cnt = 0
    #     acc = defaultdict(int)
    #     mrr = defaultdict(int)
    #     with torch.no_grad():
    #         for batch in tqdm(self.valid_loader, desc='valid'):
    #             q, q_mask, p_id, a, a_mask = batch
    #             q, q_mask = (
    #                 q.to(self.device),
    #                 q_mask.to(self.device),
    #             )
    #             q_emb = self.model(q, q_mask, "query")  # bsz x bert_dim
    #             result = self.index.search_knn(query_vectors=q_emb.cpu().numpy(), top_docs=k_max)
                
    #             for ((pred_idx_lst, _), true_idx, _a , _a_mask) in zip(result, p_id, a, a_mask):
    #                 a_len = _a_mask.sum()
    #                 _a = _a[:a_len]  # no padded values
    #                 _a = _a[1:-1]  # [CLS], [SEP]
    #                 _a_txt = self.tokenizer.decode(_a).strip()

    #                 passage = _a_txt.split("[SEP]")[-1].split("[CLS]")[-1] # title 제거
    #                 docs = [pickle.load(open(get_passage_file([idx], passages_dir=valid_dataset.passages_dir),'rb'))[idx] for idx in pred_idx_lst]

    #                 # Accuracy
    #                 for _k in k:  
    #                     if _a_txt in ' '.join(docs[:_k]): 
    #                         acc[_k] += 1
    #                     # if true_idx in pred_idx_lst[:_k]:
    #                     #     acc[_k] += 1
                    
    #                 rank = 0
    #                 # MRR
    #                 for _k in k:
    #                     try:
    #                         rank = docs[:_k].index(_a_txt) + 1
    #                         inv_rank = 1/rank
    #                     except ValueError:
    #                         inv_rank = 0
    #                     # try:
    #                     #     rank = pred_idx_lst.index(true_idx) + 1
    #                     #     inv_rank = 1/rank
    #                     # except ValueError:
    #                     #     inv_rank = 0
    #                     # print(_k, rank)
    #                     mrr[_k] += inv_rank

    #             bsz = q.size(0)
    #             sample_cnt += bsz
        
    #     acc = {_k:float(v) / float(sample_cnt) for _k, v in acc.items()}
    #     print(f"Accuracy: {acc}")
    #     mrr = {_k:float(v) / float(sample_cnt) for _k, v in mrr.items()}
    #     print(f"MRR rank: {mrr}")
        
    #     return acc, mrr


# python retriever.py -q "비밀번호를 찾고 싶어요." -k 10 > logs/clean.log
if __name__ == "__main__":
    """
    python retriever.py --article_path "dataset/baemin_qc.tsv" \
                        --model_name "model/my_model.pt" \
                        -q "비밀번호를 찾고 싶어요." \
                        -k 10
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--article_path", type=str, default="dataset/baemin_qc.tsv", required=True)
    parser.add_argument("--model_name", type=str, default="model/my_model.pt", required=True)
    # parser.add_argument("--valid_path", type=str, default=None, required=True)
    parser.add_argument("--query", "-q", type=str, required=True)
    parser.add_argument("--k", "-k", type=int, required=True)
    
    args = parser.parse_args()

    model = KobertBiEncoder()
    model.load(args.model_name)  # "model/kobert_biencoder.pt"
    model.eval()
    
    # valid_dataset = Dataset(model_name=args.model_name
    #                         data_path=args.valid_path, 
    #                         passages_dir="passages",
    #                         title_index_map_path="title_passage_map.p",
    #                         bm25=False)
    
    indexer = DenseFlatIndexer()
    indexer.deserialize(path="index", file_name=args.article_path.split('.')[-2].split('/')[-1])

    retriever = DPRRetriever(model=model, 
                             article_path='article/baemin_qc.p',
                             indexer=indexer,
                            )
    retriever.retrieve(query=args.query, k=args.k)
    # retriever.val_top_k_acc()


