# Ko-docT5query

docT5query를 한국어 데이터에 적용한 프로젝트입니다.

## 실행 방법

### 1. 가상 환경 생성

```
conda create -n doct5query python=3.10.11
conda activate doct5query
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip3 install -r requirements.txt
```

### 2. 데이터 다운로드

## 데이터 형식

### 1. 문서 (collections)

`collection.tsv`

```text
docid||text
docid||text
...
```

### 2. 질의 (queries)

`questions.tsv`

```text
qid\tquery
qid\tquery
...
```

### 3. Qrel (qrels)

`qrels.tsv`

```text
qid\tdocid
qid\tdocid
...
```

