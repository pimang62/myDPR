# Retrieval Trainer (DPR)

Retrieval 모델 훈련을 위한 Repo 

### How To Run

* Data Format for Training
  * pd.DataFrame({"title":[str], "question":[str], "content" or "asnwer":[str]})

* Data Format for Indexing
  * txt, pdf, docx anyway (now pd.DataFrame only)

1. Create vertual environment

```
conda create -n mydpr python=3.9
pip install -r requirements.txt
```

2. Chunk your dataset
- data_path : 훈련용 데이터 경로
  - title, question, content(문단) or answer(문장) column이 존재하는 dataframe 형태여야 합니다.
- passage_dir : chunked passage가 저장되는 directory
- title_index_map_path : title(key)이 같은 passage들의 index(value) hashmap
- chunk_size : 컨텍스트 윈도우
- chunk_overlap : 겹치는 정도
```
python chunker.py --data_path "dataset/path.tsv" \
                  --passages_dir "passages" \
                  --title_index_map_path "title_passage_map.p" \
                  --chunk_size 100 \
                  --chunk_overlap 0
```

3. Train
- model_name : 훈련하고 싶은 bert 모델
  - Default pre-trained 모델이 "model/kobert_biencoder.pt"입니다.
  > python model/kobert_biencoder.py 를 실행하여 얻을 수 있습니다.
- outputs_dir : trained 모델이 저장되는 path
- bm25 : True면 bm25 sampler, False면 random sampler
```
python trainer.py --model_name "model/kobert_biencoder.pt" \
                  --train_path "dataset/train_path.tsv" \
                  --valid_path "dataset/valid_path.tsv" \
                  --passages_dir "passages" \
                  --outputs_dir "model/my_model.pt" \
                  --title_index_map_path "title_passage_map.p" \
                  --bm25 False
```
OR
```
bash trainer.sh
```

cf. wandb login 계정 token 입력 > trainer.py [line 128] : wandb.init() project name과 entity 계정 이름을 변경하세요.


4. Index article for your trained model
- article_path : 새롭게 indexing하고 싶은 article
  - train시 사용한 data_path를 활용해도 OK (현재는 이렇게만 가능)
- model_name : indexing하고 싶은 trained 모델
```
python index_runner.py --article_path "dataset/path.tsv" \
                        --model_name "model/my_model.pt"
```

5. Retrieve for query using your trained model and indexed article
```
python retriever.py --article_path "dataset/path.tsv" \
                    --model_name "model/my_model.pt" \
                    -q "비밀번호를 찾고 싶어요." \
                    -k 10
```

* Output
```
문서 #37, 점수: 220.66632080078125
[CLS] 고객의 개인정보 보호 및 추가 연락 도움드립니다.[SEP][CLS] 정보통신법상 고객의 개인정보는 제공이 불가합니다. 배달 관련 사항으로 추가 연락이 필요한 경우 고객센터로 연락주시면 최대한 도움드리겠습니다.[SEP]

문서 #36, 점수: 220.27566528320312
[CLS] 리뷰 삭제 불가능, 취소 여부와 상관없이 문서 종합 제목.[SEP][CLS] 이미 작성된 리뷰는 주문이 취소되더라도 삭제되지 않습니다.[SEP]

문서 #4, 점수: 219.38571166992188
[CLS] 배민사장님광장 연동을 위한 사업자번호 및 본인인증 안내[SEP][CLS] 기존 가게와 연동을 원하는 경우, [배민사장님광장 > 배민셀프서비스] 진입 시 사업자번호 및 본인인증 절차를 통해 연결할 수 있습니다.[SEP]

문서 #90, 점수: 219.09327697753906
[CLS] 부정 클릭 방지 기능의 중요성과 효과적인 활용 방법.[SEP][CLS] 중복 클릭 방지뿐만 아니라 사용자가 비정상적인 클릭을 발생시키는 경우를 차단하여 사장님들에게 부정한 광고비가 발생되지 않도록 기능을 마련하고 있습니다. 자세한 내용은 부정 클릭 방지에 기능에 대한 회피 사례가 발생될 수 있기에 공개가 어려운 점 양해 부탁드립니다.[SEP]

문서 #20, 점수: 218.77505493164062
[CLS] 배민셀프서비스 판매 상태 및 옵션 설정 가이드[SEP][CLS] 판매 상태 변경은 [배민셀프서비스 → 상품 관리 → 판매상태 변경 / 재고 변경]에서 가능합니다. 판매상태 변경에서는 판매중인지, 판매중지 상태인지 선택할 수 있고 옵션별 판매상태도 설정할 수 있습니다. 배민셀프서비스를 통해 사장님이 직접 수정한 판매상태는 즉시 앱에 반영됩니다.[SEP]

문서 #49, 점수: 218.42617797851562
[CLS] 가게 운영과 카드 사용에 관한 법률 준수와 협조 요청 협조 요청서[SEP][CLS] 가게 운영과 관련하여 여신전문금융업법, 전자금융거래법 등을 준수하여야 합니다. 배달의민족에서 사용 가능한 카드 사용과 관련하여 카드사로부터 자료 제공 협조 요청이 있을 수 있습니다.[SEP]

문서 #42, 점수: 218.3303680419922
[CLS] 안심번호 확인과 통화 불가 상황 대응 방법 안내[SEP][CLS] 주문 전표에 기재된 안심번호로 정확하게 통화를 시도한 것이 맞는지 먼저 확인 부탁드립니다. 고객의 연락처가 결번(연락처 변경 후 수정 재등록하지 않은 상태)이거나, 착신정지된 번호일 경우 안심번호는 부여되나 전화 연결이 불가합니다. 고객과 통화가 필요한 상황일 경우 고객센터로 문의 부탁드립니다.[SEP]

문서 #94, 점수: 218.20999145507812
[CLS] 최소주문금액 및 배달 주문건 안내[SEP][CLS] 아닙니다. 최소주문금액은 배달 주문건에만 적용됩니다.[SEP]

문서 #19, 점수: 217.93743896484375
[CLS] 상품 정보 수정 요청 및 처리 절차.[SEP][CLS] 상품 정보 수정 요청은 [배민셀프서비스 → 상품 관리 → 신규 상품 등록 · 정보 수정 요청 → 수정 요청]에서 가능합니다. 단, 상품 정보 중에서는 별도의 승인 절차 없이 사장님이 앱에 바로 적용할 수 있는 직접수정 항목과 요청 내용이 승인 처리된 후 앱에 반영되는 수정요청 항목으로 구분됩니다. 정보 수정 요청은 한 번에 하나의 요청만 진행이 가능합니다. 이 점 꼭 참고 부탁

문서 #5, 점수: 217.93377685546875
[CLS] 이메일 및 휴대폰번호 변경 규정 안내[SEP][CLS] 이메일은 [배민사장님광장 > 내 정보 > 회원정보]를 통해 자유롭게 변경 가능하나, 휴대폰번호는 명의자가 동일할 경우에만 변경할 수 있습니다. 명의자가 다르거나 이름을 변경해야 할 경우, 등록된 계정 탈퇴 후 재가입을 진행해야 합니다.[SEP]```
```

### ToDo

- [ ] Extracting txt, pdf, docx

### Reference 
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
- [KorDPR](https://github.com/TmaxEdu/KorDPR)

### License 

