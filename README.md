# ``RePair``: A Toolkit for Query Refinement Gold Standard Generation
Search engines have difficulty searching into knowledge repositories since they are not tailored to the users' information needs. User's queries are, more often than not, under-specified that also retrieve irrelevant documents. Query refinement, also known as query `reformulation`, or `suggesetion`, is the process of transforming users' queries into new `refined` versions without semantic drift to enhance the relevance of search results. Prior query refiners have been benchmarked on web retrieval datasets following `weak assumptions` that users' input queries improve gradually within a search session. To fill the gap, we contribute `RePair`, an open-source configurable toolkit to generate large-scale gold-standard benchmark datasets from a variety of domains for the task of query refinement. `RePair` takes a dataset of queries and their relevance judgements (e.g. `msmarco` or `aol`), an information retrieval method (e.g., `bm25`), and an evaluation metric (e.g., `map`), and outputs refined versions of queries using a transformer (e.g., [`T5`](https://github.com/google-research/text-to-text-transfer-transformer)), each of which with the relevance improvement guarantees. Currently, `RePair` includes gold standard datasets for [`aol-ia`](https://dl.acm.org/doi/abs/10.1007/978-3-030-99736-6_42) and [`msmarco.passage`](https://www.microsoft.com/en-us/research/publication/ms-marco-human-generated-machine-reading-comprehension-dataset/).

**Future Work**: We are investigating `contexual` query refinement by incorporating query session information like user or time information of queries on the performance of neural query refinement methods compared to the lack thereof.

1. [Setup](#1-setup)
2. [Quickstart](#2-quickstart)
3. [Features](#3-features)
4. [Results](#4-results)
5. [Acknowledgement](#5-acknowledgement)
6. [License](#6-license)
7. [Citation](#7-citation)
8. [Awards](#8-awards)

## 1. Setup
You need to have ``Python=3.8`` and install [`pyserini`](https://github.com/castorini/pyserini/) package (needs `Java`), among others listed in [``requirements.txt``](requirements.txt). We also suggest you to clone our repo with the `--recurse-submodules` (altenatively, use the `git submodule update --init` inside the cloned repository) to get the trec_eval metric evaluation tool:

By ``pip``, clone the codebase and install the required packages:
```sh
git clone https://github.com/Fani-Lab/RePair
cd RePair
pip install -r requirements.txt
```

By [``conda``](https://www.anaconda.com/products/individual):

```sh
git clone https://github.com/Fani-Lab/RePair
cd RePair
conda env create -f environment.yml
conda activate repair
```
_Note: When installing `Java`, remember to set `JAVA_HOME` in Windows's environment variables._

### Lucene Indexes
To perform fast IR tasks, we need to build the indexes of document corpora or use the [`prebuilt-indexes`](https://github.com/castorini/pyserini/blob/master/docs/prebuilt-indexes.md) like [`msmarco.passage`](https://rgw.cs.uwaterloo.ca/JIMMYLIN-bucket0/pyserini-indexes/lucene-index.msmarco-v1-passage.20220131.9ea315.tar.gz). The path to the index need to be set in [`./src/param.py`](./src/param.py) like [`param.settings['msmarco.passage']['index']`](./src/param.py#L24).

In case there is no prebuilt index, steps include collecting the corpus and building an index as we did for [aol-ia](https://dl.acm.org/doi/abs/10.1007/978-3-030-99736-6_42) using [`ir-datasets`](https://ir-datasets.com/aol-ia.html).

## 2. Quickstart
We use [`T5`](https://github.com/google-research/text-to-text-transfer-transformer) to train a model, that when given an input query (origianl query), generates refined (better) versions of the query in terms of retrieving more relevant documents at higher ranking positions. We fine-tune `T5` model on `msmarco.passage` (no context) and `aol` (w/o `userid`). For `yandex` dataset, we will train `T5` from scratch since the tokens are anonymized by random numbers. 

### [`['pair']`](./src/param.py#L25)
We create training sets based on different pairings of queries and relevant passages in the `./data/preprocessed/{domain name}/` for each domain like [`./data/preprocessed/msmarco.passage/`](./data/preprocessed/msmarco.passage/) for `msmarco.passage`.

1. `ctx.query.docs`: context: query -> _concatenated_ relevant documents (passages) 
2. `ctx.docs.query`: context: _concatenated_ relevant documents (passages) -> query like in [docTTTTTTQuery](https://github.com/castorini/docTTTTTquery#learning-a-new-prediction-model-t5-training-with-tensorflow)
3. `ctx.query.doc`: context: query -> relevant document (passage)
4. `ctx.doc.query`: context: relevant documents (passages) -> query like in [docTTTTTTQuery](https://github.com/castorini/docTTTTTquery#learning-a-new-prediction-model-t5-training-with-tensorflow)

where the context will be `userid` (personalized) or empty (context free). For instance, for `msmarco.passage` which has no contextual information, we have [`docs.query`](./data/preprocessed/toy.msmarco.passage/docs.query.train.tsv) or [`query.docs`](./data/preprocessed/toy.msmarco.passage/query.docs.train.tsv) since there is no context. Further, if a query has more than one relevant documents, we can either concatenate all relevant documents into a single document, i.e., `doc`+`s` or duplicate the (query, doc) pairs for each relevant document, i.e., `doc`.

### [`['finetune']`](./src/param.py#L14)
We have used [`T5`](https://github.com/google-research/text-to-text-transfer-transformer) to generate the refinements to the original queries. We can run [`T5`](https://github.com/google-research/text-to-text-transfer-transformer) on local machine (cpu/gpu), or on google cloud (tpu), which is the [`T5`](https://github.com/google-research/text-to-text-transfer-transformer) pereferance,
1. [local machine (cpu/gpu)(`linux`, `windows`)](https://github.com/fani-lab/personalized_query_refinement/blob/main/RUNT5.md#localhost-cpu-or-gpu)
2. [`google cloud (tpu)`](https://github.com/fani-lab/personalized_query_refinement/blob/main/RUNT5.md#google-cloud-tpu)

## 3. [`['predict']`](./src/param.py#L16)
Once a transformer has been finetuned, we feed input original queries w/ or w/o context to the model and whaterver the model generates is considered as a potential refined query. To have a collection of potential refined queries for the same original query, we used the [`top-k random sampling`](https://aclanthology.org/P18-1082/) as opposed to `beam search`, suggested by [`Nogueira and Lin`](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf). So, we ran the transformer for [`nchanges`](./src/param.py#L16) times at inference and generate [`nchanges`](./src/param.py#L16) potential refined queries.

1. `ctx.query.*/pred.{refinement index}-{model checkpoint}`: query
2. `ctx.doc.*/pred.{refinement index}-{model checkpoint}`: relevant passages of a query 
3. `ctx.querydoc.*/pred.{refinement index}-{model checkpoint}`: concatenation of query and its relevant passages

We save the test file(s) as `{ctx.query.*, ctx.doc.*, ctx.querydoc.*}.test.tsv`.


## Results
We calculate the retrieval power of each query refinement on both train and test sets using IR metrics like `map` or `ndcg` compared to the original query and see if the refinements are better.

### MSMarco

### AOL
-- With User ID as context
-- Without UserID 

### Yandex

