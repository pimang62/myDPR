import os
import logging
import wandb
import torch
import transformers
from typing import Tuple, List
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import pandas as pd
from sklearn.model_selection import KFold

from encoder import KobertBiEncoder
from dpr_dataloader import Dataset, OnehotSampler, BM25Sampler, data_collator

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/log.log",
    level=logging.DEBUG,
    format="[%(asctime)s | %(funcName)s @ %(pathname)s] %(message)s",
)
logger = logging.getLogger()  # get root logger


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Trainer:
    """basic trainer"""
    def __init__(
        self,
        model,
        device,
        train_dataset,
        valid_dataset,
        num_epoch: int,
        batch_size: int,
        lr: float,
        betas: Tuple[float],
        num_warmup_steps: int,
        num_training_steps: int,
        valid_every: int,
        best_val_ckpt_path: str,
        bm25: bool
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas)
        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps, num_training_steps
        )
        self.bm25 = bm25

        if self.bm25:  # True
            self.train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset.dataset,
                batch_sampler=BM25Sampler(
                    train_dataset.dataset, batch_size=batch_size, drop_last=False
                ),
                collate_fn=lambda x: data_collator(
                    x, padding_value=train_dataset.pad_token_id
                ),
                num_workers=4,
            )
            self.valid_loader = torch.utils.data.DataLoader(
                dataset=valid_dataset.dataset,
                batch_sampler=BM25Sampler(
                    valid_dataset.dataset, batch_size=batch_size, drop_last=False
                ),
                collate_fn=lambda x: data_collator(
                    x, padding_value=valid_dataset.pad_token_id
                ),
                num_workers=4,
            )

        else:  # Random
            self.train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset.dataset,
                batch_sampler=OnehotSampler(
                    train_dataset.dataset, batch_size=batch_size, drop_last=False
                ),
                collate_fn=lambda x: data_collator(
                    x, padding_value=train_dataset.pad_token_id
                ),
                num_workers=4,
            )
            self.valid_loader = torch.utils.data.DataLoader(
                dataset=valid_dataset.dataset,
                batch_sampler=OnehotSampler(
                    valid_dataset.dataset, batch_size=batch_size, drop_last=False
                ),
                collate_fn=lambda x: data_collator(
                    x, padding_value=valid_dataset.pad_token_id
                ),
                num_workers=4,
            )

        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.valid_every = valid_every
        self.lr = lr 
        self.betas = betas 
        self.num_warmup_steps = num_warmup_steps 
        self.num_training_steps = num_training_steps
        self.best_val_ckpt_path = best_val_ckpt_path
        self.best_val_optim_path = best_val_ckpt_path.split(".pt")[0] + "_optim.pt"

        self.start_ep = 1
        self.start_step = 1

    def ibn_loss(self, pred: torch.FloatTensor):
        """in-batch negative를 활용한 batch의 loss를 계산합니다.
        pred : bsz x bsz 또는 bsz x bsz*2의 logit 값을 가짐. 후자는 hard negative를 포함하는 경우.
        """
        bsz = pred.size(0)  # 16
        target = torch.arange(bsz).to(self.device)
        if self.bm25:
            target = torch.tensor([2*i for i in range(bsz)]).to(self.device)  # 32
        return torch.nn.functional.cross_entropy(pred, target)

    def batch_acc(self, pred: torch.FloatTensor):
        """batch 내의 accuracy를 계산합니다."""
        bsz = pred.size(0)
        target = torch.arange(bsz).to(self.device)  # 주대각선이 answer
        if self.bm25:
            target = torch.tensor([2*i for i in range(bsz)]).to(self.device)
        return (pred.detach().max(-1).indices == target).sum().float() / bsz

    def fit(self):
        """모델을 학습합니다."""
        wandb.init(
            project="mydpr",
            entity="pimang62",
            config={
                "batch_size": self.batch_size,
                "lr": self.lr,
                "betas": self.betas,
                "num_warmup_steps": self.num_warmup_steps,
                "num_training_steps": self.num_training_steps,
                "valid_every": self.valid_every,
            },
        )
        logger.debug("start training")
        self.model.train()  # 학습모드
        global_step_cnt = 0
        prev_best = None
        for ep in range(self.start_ep, self.num_epoch + 1):           
            for step, batch in enumerate(
                tqdm(self.train_loader, desc=f"epoch {ep} batch"), 1
            ):
                if ep == self.start_ep and step < self.start_step:
                    continue  # 중간부터 학습시키는 경우 해당 지점까지 복원
                
                self.model.train()  # 학습 모드
                global_step_cnt += 1
                q, q_mask, _, p, p_mask = batch
                q, q_mask, p, p_mask = (
                    q.to(self.device),
                    q_mask.to(self.device),
                    p.to(self.device),
                    p_mask.to(self.device),
                )

                q_emb = self.model(q, q_mask, "query")  # (1/2)*bsz x bert_dim
                p_emb = self.model(p, p_mask, "passage")  # bsz x bert_dim

                if self.bm25:
                    pred = torch.matmul(q_emb[::2], p_emb.T)  # (1/2)*bsz x bsz
                else:
                    pred = torch.matmul(q_emb, p_emb.T)

                loss = self.ibn_loss(pred)
                acc = self.batch_acc(pred)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                log = {
                    "epoch": ep,
                    "step": step,
                    "global_step": global_step_cnt,
                    "train_step_loss": loss.cpu().item(),
                    "current_lr": float(
                        self.scheduler.get_last_lr()[0]
                    ),  # parameter group 1개이므로
                    "step_acc": acc,
                }
                if global_step_cnt % self.valid_every == 0:
                    eval_dict = self.evaluate()
                    log.update(eval_dict)
                    if (
                        prev_best is None or eval_dict["valid_loss"] < prev_best
                    ):  # best val loss인 경우 저장
                        # self.model.checkpoint(self.best_val_ckpt_path)
                        self.save_training_state(log)
                wandb.log(log)


    def evaluate(self):
        """모델을 평가합니다."""
        self.model.eval()  # 평가 모드
        loss_list = []
        sample_cnt = 0
        valid_acc = 0
        with torch.no_grad():
            for batch in self.valid_loader:
                q, q_mask, _, p, p_mask = batch
                q, q_mask, p, p_mask = (
                    q.to(self.device),
                    q_mask.to(self.device),
                    p.to(self.device),
                    p_mask.to(self.device),
                )
                q_emb = self.model(q, q_mask, "query")  # bsz x bert_dim
                p_emb = self.model(p, p_mask, "passage")  # bsz x bert_dim

                if self.bm25:
                    pred = torch.matmul(q_emb[::2], p_emb.T)  # (1/2)*bsz x bsz
                else:
                    pred = torch.matmul(q_emb, p_emb.T)
                
                loss = self.ibn_loss(pred)
                step_acc = self.batch_acc(pred)

                bsz = q.size(0)
                sample_cnt += bsz
                valid_acc += step_acc * bsz
                loss_list.append(loss.cpu().item() * bsz)
        return {
            "valid_loss": np.array(loss_list).sum() / float(sample_cnt),
            "valid_acc": valid_acc / float(sample_cnt),
        }


    def save_training_state(self, log_dict: dict) -> None:
        """모델, optimizer와 기타 정보를 저장합니다"""
        self.model.checkpoint(self.best_val_ckpt_path)
        training_state = {
            "optimizer_state": deepcopy(self.optimizer.state_dict()),
            "scheduler_state": deepcopy(self.scheduler.state_dict()),
        }
        training_state.update(log_dict)
        torch.save(training_state, self.best_val_optim_path)
        logger.debug(f"saved optimizer/scheduler state into {self.best_val_optim_path}")

    def load_training_state(self) -> None:
        """모델, optimizer와 기타 정보를 로드합니다"""
        self.model.load(self.best_val_ckpt_path)
        training_state = torch.load(self.best_val_optim_path)
        logger.debug(
            f"loaded optimizer/scheduler state from {self.best_val_optim_path}"
        )
        self.optimizer.load_state_dict(training_state["optimizer_state"])
        self.scheduler.load_state_dict(training_state["scheduler_state"])
        self.start_ep = training_state["epoch"]
        self.start_step = training_state["step"]
        logger.debug(
            f"resume training from epoch {self.start_ep} / step {self.start_step}"
        )


if __name__ == "__main__":
    """
    python trainer.py --model_name "model/kobert_biencoder.pt" \
                      --train_path "dataset/train_baemin_qc.tsv" \
                      --valid_path "dataset/valid_baemin_qc.tsv" \
                      --passages_dir "passages" \
                      --outputs_dir "model/my_model.pt" \
                      --title_index_map_path "title_passage_map.p" \
                      --bm25 False
    """
    import argparse
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--model_name", type=str, default="model/kobert_biencoder.pt", required=True)
    # parser.add_argument("--data_path", type=str, default="dataset/baemin_qc.tsv", required=True)
    parser.add_argument("--train_path", type=str, default="dataset/train_baemin_qc.tsv", required=True)
    parser.add_argument("--valid_path", type=str, default="dataset/valid_baemin_qc.tsv", required=True)
    parser.add_argument("--passages_dir", type=str, default="passages", required=True)
    parser.add_argument("--outputs_dir", type=str, default="model/my_model.pt", required=True)
    parser.add_argument("--title_index_map_path", type=str, default="title_passage_map.p", required=True)
    parser.add_argument("--bm25", type=str2bool, default=False, required=True)

    args = parser.parse_args()
    
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = KobertBiEncoder()
    model.load(model_ckpt_path=args.model_name)
    
    ## KFold split
    # kf = KFold(n_splits=2, shuffle=False)  # 10
    # total_data = pd.read_csv(args.data_path, sep='\t')

    # for i, (t, v) in enumerate(kf.split(total_data), start=1):
    #     print(f"Fold {i}:")
    #     train_data, valid_data = total_data.iloc[t], total_data.iloc[v]
    #     train_data.to_csv(f"dataset/train_data_{i}fold.tsv", sep='\t')
    #     valid_data.to_csv(f"dataset/valid_data_{i}fold.tsv", sep='\t')
    #     train_path, valid_path \
    #         = f"dataset/train_data_{i}fold.tsv", f"dataset/valid_data_{i}fold.tsv"

    train_dataset = Dataset(model_name=args.model_name,
                            data_path=args.train_path,  # train_path
                            passages_dir=args.passages_dir,
                            title_index_map_path=args.title_index_map_path,
                            bm25=args.bm25)  # False: Random sampling
    valid_dataset = Dataset(model_name=args.model_name,
                            data_path=args.valid_path,  # valid_path
                            passages_dir=args.passages_dir,
                            title_index_map_path=args.title_index_map_path,
                            bm25=args.bm25)  # False: Random sampling
    my_trainer = Trainer(
        model=model,
        device=device,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        num_epoch=50,
        batch_size=32,  # 128 - 32
        lr=1e-6,
        betas=(0.9, 0.99),
        num_warmup_steps=1000,
        num_training_steps=100000,
        valid_every=30,
        best_val_ckpt_path=args.outputs_dir,  # 3/28 여기까지 돌렸음
        bm25=args.bm25  # False: Random sampling
    )
    my_trainer.fit()
    eval_dict = my_trainer.evaluate()
    print(eval_dict)
