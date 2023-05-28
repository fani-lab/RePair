# ``RePair``: A Toolkit for Query Refinement Gold Standard Generation Using Transformers
Search engines have difficulty searching into knowledge repositories since they are not tailored to the users' information needs. User's queries are, more often than not, under-specified that also retrieve irrelevant documents. Query refinement, also known as query `reformulation`, or `suggesetion`, is the process of transforming users' queries into new `refined` versions without semantic drift to enhance the relevance of search results. Prior query refiners have been benchmarked on web retrieval datasets following `weak assumptions` that users' input queries improve gradually within a search session. To fill the gap, we contribute `RePair`, an open-source configurable toolkit to generate large-scale gold-standard benchmark datasets from a variety of domains for the task of query refinement. `RePair` takes a dataset of queries and their relevance judgements (e.g. `msmarco` or `aol`), an information retrieval method (e.g., `bm25`), and an evaluation metric (e.g., `map`), and outputs refined versions of queries using a transformer (e.g., [`T5`](https://github.com/google-research/text-to-text-transfer-transformer)), each of which with the relevance improvement guarantees. Currently, `RePair` includes gold standard datasets for [`aol-ia`](https://dl.acm.org/doi/abs/10.1007/978-3-030-99736-6_42) and [`msmarco.passage`](https://www.microsoft.com/en-us/research/publication/ms-marco-human-generated-machine-reading-comprehension-dataset/).

**Future Work**: We are investigating `contexual` query refinement by incorporating query session information like user or time information of queries on the performance of neural query refinement methods compared to the lack thereof.

<table align="center" border=0>
<tr>
<td colspan=2>

- [1. Setup](#1-setup)
  * [Lucene Indexes](#lucene-indexes)
- [2. Quickstart](#2-quickstart)
  * [`pair`](#pair)
  * [`finetune`](#finetune)
  * [`predict`](#predict)
  * [`search`](#search)
  * [`eval`](#eval)
  * [`agg, box`](#agg-box)
  * [`dense_retrieve`](#dense_retrieve)
- [3. Gold Standard Datasets](#3-gold-standard-datasets)
  * [File Structure](#file-structure)
  * [Settings](#settings)
  * [Stats](#stats)
  * [Samples](#samples)
- [4. Acknowledgement](#4-acknowledgement)
- [5. License](#5-license)

</td>
<td ><img src='./misc/flow.png' width="50%" /></td>
<td ><img src='./misc/class.png' width="100%" /></td>
</tr>
</table>

## 1. Setup
You need to have ``Python=3.8`` and install [`pyserini`](https://github.com/castorini/pyserini/) package (needs `Java`), among others listed in [``requirements.txt``](requirements.txt). We also suggest you to clone our repo with the `--recurse-submodules` (altenatively, use the `git submodule update --init` inside the cloned repository) to get the [`trec_eval`](https://github.com/usnistgov/trec_eval) metric evaluation tool:

By ``pip``, clone the codebase and install the required packages:
```sh
git clone https://github.com/Fani-Lab/RePair --recurse-submodules
cd RePair
pip install -r requirements.txt
```

By [``conda``](https://www.anaconda.com/products/individual):

```sh
git clone https://github.com/Fani-Lab/RePair --recurse-submodules
cd RePair
conda env create -f environment.yml
conda activate repair
```
_Note: When installing `Java`, remember to set `JAVA_HOME` in Windows's environment variables._

For [`trec_eval`](https://github.com/usnistgov/trec_eval):
```sh
cd src/trec_eval.9.0.4
make 
cd ..
```

### Lucene Indexes
To perform fast IR tasks, we need to build the indexes of document corpora or use the [`prebuilt-indexes`](https://github.com/castorini/pyserini/blob/master/docs/prebuilt-indexes.md) like [`msmarco.passage`](https://rgw.cs.uwaterloo.ca/JIMMYLIN-bucket0/pyserini-indexes/lucene-index.msmarco-v1-passage.20220131.9ea315.tar.gz). The path to the index need to be set in [`./src/param.py`](./src/param.py) like [`param.settings['msmarco.passage']['index']`](./src/param.py#L24).

In case there is no prebuilt index, steps include collecting the corpus and building an index as we did for [`aol-ia`](https://dl.acm.org/doi/abs/10.1007/978-3-030-99736-6_42) using [`ir-datasets`](https://ir-datasets.com/aol-ia.html).

## 2. Quickstart
We use [`T5`](https://github.com/google-research/text-to-text-transfer-transformer) to train a model, that when given an input query (origianl query), generates refined (better) versions of the query in terms of retrieving more relevant documents at higher ranking positions. Currently, we finetuned [`T5`](https://github.com/google-research/text-to-text-transfer-transformer) model on `msmarco.passage` (no context) and `aol` (w/o `userid`). For `yandex` dataset, we will train [`T5`](https://github.com/google-research/text-to-text-transfer-transformer) from scratch since the tokens are anonymized by random numbers. 

As seen in the above [`workflow`](./misc/workflow.png), `RePair` has four pipelined steps: 
> 1. Transformer Finetuning: [`pair`, `finetune`]
> 2. Refined Query Prediction: [`predict`]
> 3. Performance Evaluation: [`search`, `eval`]
> 4. Dataset Curation: [`agg`, `box`]

To run `RePair` pipeline, we need to set the required parameters of each step in [`./src/param.py`](./src/param.py) such as pairing strategy ([`pairing`](./src/param.py#L28)) for a query set, the choice of transformer ([`t5model`](./src/param.py#L14)), and etc. Then, the pipeline can be run by its driver at [`./src/main.py`](./src/main.py):

```sh
python -u main.py -data ../data/raw/toy.msmarco.passage -domain msmarco.passage
```
```sh
python -u main.py -data ../data/raw/toy.aol-ia -domain aol-ia
```
```sh
python -u main.py -data ../data/raw/toy.msmarco.passage ../data/raw/toy.aol-ia -domain msmarco.passage aol-ia
```

### [`['pair']`](./src/param.py#L25)
We create training sets based on different pairings of queries and relevant passages in the [`./data/preprocessed/{domain name}/`](./data/preprocessed/) for each domain like [`./data/preprocessed/toy.msmarco.passage/`](./data/preprocessed/toy.msmarco.passage/) for `msmarco.passage`.

1. `ctx.query.docs`: context: query -> _concatenated_ relevant documents (passages) 
2. `ctx.docs.query`: context: _concatenated_ relevant documents (passages) -> query, like in [docTTTTTTQuery](https://github.com/castorini/docTTTTTquery#learning-a-new-prediction-model-t5-training-with-tensorflow)
3. `ctx.query.doc`: context: query -> relevant document (passage)
4. `ctx.doc.query`: context: relevant documents (passages) -> query

where the context will be `userid` (personalized) or empty (context free). For instance, for `msmarco.passage` which has no contextual information, we have [`docs.query`](./data/preprocessed/toy.msmarco.passage/docs.query.passage.train.tsv) or `query.docs` since there is no context. Further, if a query has more than one relevant documents, we can either _concatenate_ all relevant documents into a single document, i.e., `doc`+`s` or _duplicate_ the (query, doc) pairs for each relevant document, i.e., `doc`.

After this step, [`./data/`](./data) directory looks like:

```bash
├── data
│   ├── preprocessed
│   │   └── toy.msmarco.passage
│   │       ├── docs.query.passage.test.tsv
│   │       ├── docs.query.passage.train.tsv
│   │       ├── queries.qrels.docs.ctx.passage.test.tsv
│   │       └── queries.qrels.docs.ctx.passage.train.tsv
```

### [`['finetune']`](./src/param.py#L14)
We have used [`T5`](https://github.com/google-research/text-to-text-transfer-transformer) to generate the refinements to the original queries. We can run [`T5`](https://github.com/google-research/text-to-text-transfer-transformer) on local machine (cpu/gpu), or on google cloud (tpu), which is the [`T5`](https://github.com/google-research/text-to-text-transfer-transformer) pereferance,
> - [`local machine (cpu, gpu)(linux, windows)`](https://github.com/fani-lab/personalized_query_refinement/blob/main/RUNT5.md#localhost-cpu-or-gpu)
> - [`google cloud (tpu)`](https://github.com/fani-lab/personalized_query_refinement/blob/main/RUNT5.md#google-cloud-tpu)

We store the finetuned transformer in `./output/{domain name}/{transformer name}.{pairing strategy}`. For instance, for  [`T5`](https://github.com/google-research/text-to-text-transfer-transformer) whose `small` version has been finetuned on a local machine for `toy.msmarco.passage`, we save the model in [`./output/toy.msmarco.passage/t5.small.local.docs.query.passage/`](./output/toy.msmarco.passage/t5.small.local.docs.query.passage/)

After this step, [`./output/`](./output) looks like:

```bash
├── output
│   ├── t5-data
│   │   ├── pretrained_models
│   │   │   └── small
│   │   └── vocabs
│   │       ├── cc_all.32000
│   │       └── cc_en.32000
│   └── toy.msmarco.passage
│       └── t5.small.local.docs.query.passage
│           ├── checkpoint
│           ├── events.out.tfevents.1675961098.HFANI
│           ├── graph.pbtxt
│           ├── model.ckpt-1000000.data-00000-of-00002
│           ├── model.ckpt-1000000.index
│           ├── model.ckpt-1000000.meta
│           ├── model.ckpt-1000005.data-00000-of-00002
│           ├── model.ckpt-1000005.index
│           ├── model.ckpt-1000005.meta
│           ├── operative_config.gin
```

### [`['predict']`](./src/param.py#L16)
Once a transformer has been finetuned, we feed input original queries w/ or w/o context to the model and whaterver the model generates is considered as a `potential` refined query. To have a collection of potential refined queries for the same original query, we used the [`top-k random sampling`](https://aclanthology.org/P18-1082/) as opposed to `beam search`, suggested by [`Nogueira and Lin`](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf). So, we ran the transformer for [`nchanges`](./src/param.py#L16) times at inference and generate [`nchanges`](./src/param.py#L16) potential refined queries. 

We store the `i`-th potential refined query of original queries at same folder as the finetuned model, i.e., `./output/{domain name}/{transformer name}.{pairing strategy}/pred.{refinement index}-{model checkpoint}` like [`./output/toy.msmarco.passage/t5.small.local.docs.query.passage/pred.0-1000005`](./output/toy.msmarco.passage/t5.small.local.docs.query.passage/pred.0-1000005)

After this step, prediction files will be added to [`./output`](./output):

```bash
├── output
│   └── toy.msmarco.passage
│       └── t5.small.local.docs.query.passage
│           ├── original.-1
│           ├── pred.0-1000005
│           ├── pred.1-1000005
│           ├── pred.2-1000005
│           ├── pred.3-1000005
│           ├── pred.4-1000005
```

### [`['search']`](./src/param.py#L17)
We search the relevant documents for both the original query and each of the `potential` refined queries. We need to set an information retrieval method, called ranker, that retrieves relevant documents and ranks them based on relevance scores. We integrate [`pyserini`](https://github.com/castorini/pyserini), which provides efficient implementations of sparse and dense rankers, including `bm25` and `qld` (query likelihood with Dirichlet smoothing). 

We store the result of search for the `i`-th potential refined query at same folder in files with names ending with ranker, i.e., `./output/{domain name}/{transformer name}.{pairing strategy}/pred.{refinement index}-{model checkpoint}.{ranker name}` like [`./output/toy.msmarco.passage/t5.small.local.docs.query.passage/pred.0-1000005.bm25`](./output/toy.msmarco.passage/t5.small.local.docs.query.passage/pred.0-1000005.bm25).

After this step, search results will be added to [`./output`](./output):

```bash
├── output
│   └── toy.msmarco.passage
│       └── t5.small.local.docs.query.passage
│           ├── original.-1.bm25
│           ├── original.-1.qld
│           ├── pred.0-1000005.bm25
│           ├── pred.0-1000005.qld
│           ├── pred.1-1000005.bm25
│           ├── pred.1-1000005.qld
│           ├── pred.2-1000005.bm25
│           ├── pred.2-1000005.qld
│           ├── pred.3-1000005.bm25
│           ├── pred.3-1000005.qld
│           ├── pred.4-1000005.bm25
│           ├── pred.4-1000005.qld
```

### [`['eval']`](./src/param.py#L20)
The search results of each potential refined queries are evaluated based on how they improve the performance with respect to an evaluation metric like `map` or `mrr`. 

We store the result of evaluation for the `i`-th potential refined query at same folder in files with names ending with evaluation metric, i.e., `./output/{domain name}/{transformer name}.{pairing strategy}/pred.{refinement index}-{model checkpoint}.{ranker name}.{metric name}` like [`./output/toy.msmarco.passage/t5.small.local.docs.query.passage/pred.0-1000005.bm25.map`](./output/toy.msmarco.passage/t5.small.local.docs.query.passage/pred.0-1000005.bm25.map).


After this step, evaluation results will be added to [`./output`](./output):

```bash
├── output
│   └── toy.msmarco.passage
│       └── t5.small.local.docs.query.passage
│           ├── original.-1.bm25.map
│           ├── original.-1.bm25.success.10
│           ├── original.-1.qld.map
│           ├── original.-1.qld.success.10
```

### [`['agg', 'box']`](./src/param.py#L12)
Finaly, we keep those potential refined queries whose performance (metric score) have been better or equal compared to the original query, i.e., `refined_query_metric >= original_query_metric and refined_q_metric > 0`.

We keep two main datasets as the final outcome of the `RePair` pipeline:

> 1. `./output/{input query set}/{transformer name}.{pairing strategy}/{ranker}.{metric}.agg.gold.tsv`: contains the original queries and their refined queries that garanteed the `better` performance along with the performance metric values

> 2. `./output/{input query set}/{transformer name}.{pairing strategy}/{ranker}.{metric}.agg.all.tsv`: contains the original queries and `all` their predicted refined queries along with the performance metric values

For instance, for `toy` query sets of `msmarco.passage` and `aol-ia.title`, here are the files:

[`./output/toy.msmarco.passage/t5.small.local.docs.query.passage/bm25.map.agg.gold.tsv`](./output/toy.msmarco.passage/t5.small.local.docs.query.passage/bm25.map.agg.gold.tsv)

[`./output/toy.msmarco.passage/t5.small.local.docs.query.passage/bm25.map.agg.all.tsv`](./output/toy.msmarco.passage/t5.small.local.docs.query.passage/bm25.map.agg.all.tsv)

[`./output/toy.aol-ia/t5.small.local.docs.query.title/bm25.map.agg.gold.tsv`](./output/toy.aol-ia/t5.small.local.docs.query.title/bm25.map.agg.gold.tsv)

[`./output/toy.aol-ia/t5.small.local.docs.query.title/bm25.map.agg.all.tsv`](./output/toy.aol-ia/t5.small.local.docs.query.title/bm25.map.agg.all.tsv)

For boxing, since we keep the performances for all the potential queries, we can change the condition and have a customized selection like having [`diamond`](https://dl.acm.org/doi/abs/10.1145/3459637.3482009) refined queries with maximum possible performance, i.e., `1` by setting the condition: `refined_query_metric >= original_query_metric and refined_q_metric == 1`. The boxing condition can be set at [`./src/param.py`](./src/param.py#L12). 

```
'box': {'gold':     'refined_q_metric >= original_q_metric and refined_q_metric > 0',
        'platinum': 'refined_q_metric > original_q_metric',
        'diamond':  'refined_q_metric > original_q_metric and refined_q_metric == 1'}
```


After this step, [`./output`](./output) will further include:

```bash
├── output
│   └── toy.msmarco.passage
│       └── t5.small.local.docs.query.passage
│           ├── qld.map.agg.all.tsv
│           ├── qld.map.agg.all_.tsv
│           ├── qld.map.agg.gold.tsv
│           ├── qld.map.boxes
│           │   ├── diamond.tsv
│           │   ├── gold.tsv
│           │   ├── platinum.tsv
│           │   └── stamps
│           │       ├── diamond.change.qld.map
│           │       ├── diamond.original.qld.map
│           │       ├── gold.change.qld.map
│           │       ├── gold.original.qld.map
│           │       ├── platinum.change.qld.map
│           │       └── platinum.original.qld.map
│           ├── qld.success.10.agg.all.tsv
│           ├── qld.success.10.agg.all_.tsv
│           ├── qld.success.10.agg.gold.tsv
│           └── qld.success.10.boxes
│               ├── diamond.tsv
│               ├── gold.tsv
│               ├── platinum.tsv
│               └── stamps
│                   ├── diamond.change.qld.success.10
│                   ├── diamond.original.qld.success.10
│                   ├── gold.change.qld.success.10
│                   ├── gold.original.qld.success.10
│                   ├── platinum.change.qld.success.10
│                   └── platinum.original.qld.success.10
```


### [`['dense_retrieve']`](./src/param.py#L17)

It is imperative for us to support dense retrieval as a ranker method for which we choose to explore `tct_colbert` the current `state-of-the-art` method.

`tct colbert` is supported directly by `pyserini` , which allows us to index and search.

We explain in detail how to do this for any dataset and release our dense index for `aol.title` and `aol.title.url`, for `msmarco.passage`, we used [the prebuild index](https://github.com/castorini/pyserini/blob/master/docs/experiments-tct_colbert-v2.md) provided by pyserini 

**Indexing**:
Before building a dense index, we need to encode the passages either in faiss format or jsonl format. We provide a sample command that indexes `aol.title` :
```
python -m pyserini.encode \
  input   --corpus  data/raw/aol-ia/title\
          --fields text \  
          --delimiter "\n" \
          --shard-id 0 \   
          --shard-num 1 \  
  output  --embeddings data/raw/aol-ia/dense-encoder/tct_colbert.title \
          --to-faiss \
  encoder --encoder castorini/tct_colbert-v2-hnp-msmarco \
          --fields text \  
          --batch 32 
```
Now once the encoding is done, we need to index this, This indexing will take about 24 hours:
```
    python -m pyserini.index.faiss \
  --input  data/raw/aol-ia/dense-encoder/tct_colbert.title \  
  --output data/raw/aol-ia/dense-index/tct_colbert.title \
  --hnsw
```
We chose to go with the HNSW file structure,you can explore more about it [here](https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexHNSW.html#struct-faiss-indexhnsw)

[**Searching**:](./src/dal/ds.py#L53)
We added the searching to our pipeline, since it's very similar as searching in a sparse index. 
```python
from pyserini.search.faiss import TctColBertQueryEncoder, FaissSearcher
encoder = TctColBertQueryEncoder('./data/raw/aol-ia/dense-encoder/tct_colbert.title')
searcher = FaissSearcher('./data/raw/aol-ia/dense-index/tct_colbert.title',encoder)
```
We do not apply `tct_colbert` to our whole dataset instead we attempt to improve our predicted queries which does not meet a certain condition:
``` 
original_q_metric > refined_q_metric and 0 >= original_q_metric >= 1
```

After this step, the [`./output`](./output) folder will have a final structure as below:

```bash
├── output
│   └── toy.msmarco.passage
│       └── t5.small.local.docs.query.passage
│           ├── qld.map.agg.all.tsv
│           ├── qld.map.agg.all_.tsv
│           ├── qld.map.agg.gold.tsv
|           ├── qld.map.agg.no_pred.tsv
│           ├── qld.recip_rank.10.agg.all.tsv
│           ├── qld.recip_rank.10.agg.all_.tsv
|           ├── qld.recip_rank.agg.no_pred.tsv
|           ├── original.no_pred.tct_colbert
|           ├── original.no_pred.tct_colbert.map
|           ├── original.no_pred.tct_colbert.recip_rank.10
|           ├── pred.no_pred.tct_colbert
|           ├── pred.no_pred.tct_colbert.map
|           ├── pred.no_pred.tct_colbert.recip_rank.10
|           ├── colbert.comparison.no_pred.map.tsv
|           ├── colbert.comparison.no_pred.recip_rank.10.tsv
│           ├── qld.map.boxes
│           │   ├── diamond.tsv
│           │   ├── gold.tsv
│           │   ├── platinum.tsv
│           │   └── stamps
│           │       ├── diamond.change.qld.map
│           │       ├── diamond.original.qld.map
│           │       ├── gold.change.qld.map
│           │       ├── gold.original.qld.map
│           │       ├── platinum.change.qld.map
│           │       └── platinum.original.qld.map
│           ├── qld.success.10.agg.all.tsv
│           ├── qld.success.10.agg.all_.tsv
│           ├── qld.success.10.agg.gold.tsv
│           └── qld.success.10.boxes
│               ├── diamond.tsv
│               ├── gold.tsv
│               ├── platinum.tsv
│               └── stamps
│                   ├── diamond.change.qld.success.10
│                   ├── diamond.original.qld.success.10
│                   ├── gold.change.qld.success.10
│                   ├── gold.original.qld.success.10
│                   ├── platinum.change.qld.success.10
│                   └── platinum.original.qld.success.10
```


## 3. Gold Standard Datasets 

| query set | final gold standard dataset | data folder (raw and preprocessed) | output folder (model, predictions, ...) | 
|:---:|:---:|:---:|:---:|
| `msmarco.passage` | [`./output/msmarco.passage/t5.base.gc.docs.query.passage/bm25.map.agg.gold.tsv`](https://uwin365-my.sharepoint.com/:u:/g/personal/lakshmiy_uwindsor_ca/EeMQjTbagV9GplPakERqywYBZqBB6xkJzCXfmYQnS5FABw?e=b9bKRJ) 398 MB | [`./data/raw/msmarco.passage`](https://uwin365-my.sharepoint.com/:f:/g/personal/lakshmiy_uwindsor_ca/EuH9N7rt8CRAhowJbK2CZzUBoWNrzP3sh2ErhavF5p534w?e=p5hvE5)<br>[`./data/preprocessed/msmarco.passage`](https://uwin365-my.sharepoint.com/:f:/g/personal/lakshmiy_uwindsor_ca/EtQG-QohySlAleQ6caEyyTYB3xsCQ3tTnYHTBIj5-fFnFQ?e=8K6Ce8) | [`./output/msmarco.passage/t5.base.gc.docs.query.passage/`](https://uwin365-my.sharepoint.com/:f:/g/personal/lakshmiy_uwindsor_ca/Enf3gIQZIeBNlgmyWXqob1EBgY7zVZpYagWTFX8JrGe98g?e=YPqYgz) | 
| `aol-ia.title` | [`./output/aol-ia/t5.base.gc.docs.query.title/bm25.map.agg.gold.tsv`](https://uwin365-my.sharepoint.com/:u:/g/personal/lakshmiy_uwindsor_ca/EVkDvYIyWyFGjEl88GAcKXABKVWSGITtOA8EEBeFAmc9Zw?e=bq3Ydd) <br> 756 MB| [`./data/raw/aol-ia`](https://uwin365-my.sharepoint.com/:f:/g/personal/lakshmiy_uwindsor_ca/EiYYSmz-L-VHqj8x-Zl58LIBl1XKzmgI6hmZHz8rruMfeA?e=VTDsvC) <br>[`./data/preprocessed/aol-ia`](https://uwin365-my.sharepoint.com/:f:/g/personal/lakshmiy_uwindsor_ca/EqGqoB05KMdAn0j1frOkIV4BS2cE7bWwbSysVXtxkiSNrA?e=mf8loW) | [`./output/aol-ia/t5.base.gc.docs.query.title/`](https://uwin365-my.sharepoint.com/:f:/g/personal/lakshmiy_uwindsor_ca/EunaA74D03tMk21oGHlWFccBZBzQuhFCE7J21BRuIUu4Dw?e=XfkcoO) | 
| `aol-ia.url.title` | [`./output/aol-ia/t5.base.gc.docs.query.url.title/bm25.map.agg.gold.tsv`](https://uwin365-my.sharepoint.com/:u:/g/personal/lakshmiy_uwindsor_ca/Eaf9S3WqvaBNlzu1MZhB8ZwBwFUeLkiKWumy-VNbej_Iqw?e=OwWnEy) <br> 706 MB | [`./data/raw/aol-ia`](https://uwin365-my.sharepoint.com/:f:/g/personal/lakshmiy_uwindsor_ca/EiYYSmz-L-VHqj8x-Zl58LIBl1XKzmgI6hmZHz8rruMfeA?e=VTDsvC) <br>[`./data/preprocessed/aol-ia`](https://uwin365-my.sharepoint.com/:f:/g/personal/lakshmiy_uwindsor_ca/EqGqoB05KMdAn0j1frOkIV4BS2cE7bWwbSysVXtxkiSNrA?e=mf8loW) | [`./output/aol-ia/t5.base.gc.docs.query.url.title/`](https://uwin365-my.sharepoint.com/:f:/g/personal/lakshmiy_uwindsor_ca/ErGC88ga8VVMj9z49C96pgsBw3l5W6O3px680-ElvyaTWw?e=Fek7LD) | 

### File Structure
Here is the refined queries for the two original queries in [`./output/aol-ia/t5.base.gc.docs.query.title/bm25.map.agg.gold.tsv`](https://uwin365-my.sharepoint.com/:u:/g/personal/lakshmiy_uwindsor_ca/EVkDvYIyWyFGjEl88GAcKXABKVWSGITtOA8EEBeFAmc9Zw?e=bq3Ydd) for `aol-ia.title`:

```
qid	        order	query	                        bm25.map
8c418e7c9e5993	-1	rentdirect com	                0.0
8c418e7c9e5993	pred.9	hurston apartments	        0.0556
8c418e7c9e5993	pred.4	rental apartments	        0.0357
8c418e7c9e5993	pred.8	apartments nyc	                0.0312
8c418e7c9e5993	pred.6	first class apartments in nyc	0.0135
8c418e7c9e5993	pred.2	apartments and new york	        0.0068
0cc411681d1441	-1	staple com	                0.037
0cc411681d1441	pred.8	staple pubs	                1.0
0cc411681d1441	pred.7	staple england pub	        0.5
0cc411681d1441	pred.1	staple east of england	        0.1
0cc411681d1441	pred.10	staple	                        0.0385
0cc411681d1441	pred.3	staple england	                0.0385
0cc411681d1441	pred.4	staple england	                0.0385
0cc411681d1441	pred.6	staple england	                0.0385
```
As seen, `order: -1` shows the original query with its retrieval preformance. For the rest, it shows the refined queries in decreasing retrieval performance. For instance, for the original query `query: staple com`, the retrieval performance is `bm25.map: 0.037` while the best refined query could imporove it to `bm25.map: 1.0`! 

### Settings

`RePair` has generated gold standard query refinement datasets for `msmarco.passage` and `aol-ia` query sets using `t5.base` transformer on google cloud's tpus (`gc`) with `docs.query` pairing strategy for `bm25` ranker and `map` evaluation metric. The golden datasets along with all the artifacts including preprocessed `docs.query` pairs, model checkpoint, predicted refined queries, their search results and evaluation metric values are available at the above links. The running settings were:

```
settings = {
    't5model': 'base.gc'
    'iter': 4000,       # number of finetuning iteration for t5
    'nchanges': 10,     # number of changes to a query
    'ranker': 'bm25',   
    'batch': 100,       # search per batch of queries using pyserini, if None, search per query
    'topk': 100,        # number of retrieved documents for a query
    'large_ds': True,   # set if dataset size is greater than one Million
    'metric': 'map',    # any valid trec_eval.9.0.4 metric like map, ndcg, recip_rank, ...
    'box': {'gold':     'refined_q_metric >= original_q_metric and refined_q_metric > 0',
            'platinum': 'refined_q_metric > original_q_metric',
            'diamond':  'refined_q_metric > original_q_metric and refined_q_metric == 1'},
    'msmarco.passage': {
        'index': '../data/raw/msmarco.passage/lucene-index.msmarco-v1-passage.20220131.9ea315/',
        'pairing': [None, 'docs', 'query'],     # input=doc(s), output=query, s means concat of relevant docs
        'lseq':{"inputs": 32, "targets": 256},  # query length and doc length for t5 model,
    },
    'aol-ia': {
        'index_item': ['title'], # ['url'], ['title', 'url'], ['title', 'url', 'text']
        'index': '../data/raw/aol-ia/lucene-index/title/',
        'pairing': [None, 'docs', 'query'], #input=doc(s) output=query
        'lseq':{"inputs": 32, "targets": 256},  # query length and doc length for t5 model,
        'filter': {'minql': 1, 'mindocl': 10}.  # [min query length, min doc length], after merge queries with relevant 'index_item', if |query| <= minql drop the row, if |'index_item'| < mindocl, drop row
    }
}
```

### Stats

| query set={q}   |    #q     |  avg\|q\| | avg`bm25.map`(q) |   #gold   | avg\|q*\| |  %  | avg`bm25.map`(q*) |    Δ%    | #diamond (ap=1)  | %q* |
|-----------------|:---------:|:---------:|:------:|:---------:|:----------:|:---:|:------:|:--------:|:---------------:|:----------------:|
| msmarco.passage |  502,939  |   5.9675  | 0.0862 |  414,337  |   7.4419   | 82% | 0.5704 |  +562 %  |     176,922     |        35%       |
| aol-ia-title    | 4,459,613 |   3.5849  | 0.0252 | 2,583,023 |   3.1270   | 58% | 0.4175 | +1,556 % |     649,764     |        14%       |
| aol-ia-url-title| 4,672,506 |   3.5817  | 0.0271 | 2,421,347 |   3.5354   | 52% | 0.3997 | +1,374 % |     591,001     |        13%       |

### Samples

![image](https://user-images.githubusercontent.com/8619934/221714147-d2c212b8-8e00-47b4-8f3f-89fe805d441a.png)

## 4. Acknowledgement
We benefit from [``trec_eval``](https://github.com/usnistgov/trec_eval), [``pyserini``](https://github.com/castorini/pyserini), [``ir-dataset``](https://ir-datasets.com/), and other libraries. We would like to thank the authors of these libraries and helpful resources.
  
## 5. License
©2023. This work is licensed under a [CC BY-NC-SA 4.0](license.txt) license.

Yogeswar Lakshmi Narayanan<sup>1</sup>, [Hossein Fani](https://hosseinfani.github.io/)<sup>1,2</sup> 

<sup><sup>1</sup>School of Computer Science, Faculty of Science, University of Windsor, ON, Canada.</sup>
<sup><sup>2</sup>[hfani@uwindsor.ca](mailto:hfani@uwindsor.ca)</sup>

