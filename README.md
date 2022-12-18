# Personalized Query Refinement
Creating alternative queries, also known as query reformulation, has been shown to improve users' search experience.  Recently, neural transformers like `(* * * )` have shown increased effectiveness when applied for learning to reformulate an original query into effective alternate queries. Such methods however forgo incorporating user information. We investigate personalized query reformulation by incorporating user information on the performance of neural query reformulation methods compared to the lack thereof on `aol` and `yandex` query search logs. Our experiments demonstrate the synergistic effects of taking user information into account for query reformulation.


## Installation
You need to have ``Python=3.8`` and install [`pyserini`](https://github.com/castorini/pyserini/) package (needs `Java`), among others listed in [``requirements.txt``](requirements.txt):

By ``pip``, clone the codebase and install the required packages:
```sh
git clone https://github.com/Fani-Lab/personalized_query_refinement
cd personalized_query_refinement
pip install -r requirements.txt
```

By [``conda``](https://www.anaconda.com/products/individual):

```sh
git clone https://github.com/Fani-Lab/personalized_query_refinement
cd personalized_query_refinement
conda env create -f environment.yml
conda activate pqr
```
_Note: When installing `Java`, remember to set `JAVA_HOME` in Windows's environment variables._

### Lucene Indexes
To perform fast IR tasks, we need to build the indexes of document corpora or use the [`prebuilt-indexes`](https://github.com/castorini/pyserini/blob/master/docs/prebuilt-indexes.md) like [`msmarco-passage`](https://rgw.cs.uwaterloo.ca/JIMMYLIN-bucket0/pyserini-indexes/lucene-index.msmarco-v1-passage.20220131.9ea315.tar.gz). The path to the index need to be set in [`./src/param.py`](./src/param.py) like [`param.settings['msmarco-passage']['index']`](https://github.com/fani-lab/personalized_query_refinement/blob/main/src/param.py#L17).

## Model Training and Test
We use [`T5`](https://github.com/google-research/text-to-text-transfer-transformer) to train a model, that when given an input query (origianl query), generates refined (better) versions of the query in terms of retrieving more relevant documents at higher ranking positions. We fine-tune `T5` model on `msmarco` (w/o `userid`) and `aol` (w/ and w/o `userid`). For `yandex` dataset, we need to train `T5` from scratch since the tokens are anonymized by random numbers. 

`check [T5X](https://github.com/google-research/t5x)`

### Dataset
We create training sets based on different pairings of queries and relevant passages in the `./data/preprocessed/{domain name}/` for each domain like [`./data/preprocessed/msmarco/`](./data/preprocessed/msmarco/) for `msmarco`.

1. `ctx.query.doc`: context: query -> relevant passages
2. `ctx.doc.query`: context: relevant passages -> queries like in [docTTTTTTQuery](https://github.com/castorini/docTTTTTquery#learning-a-new-prediction-model-t5-training-with-tensorflow)

where the context will be `userid` (personalized) or empty (context free). For instance, for `msmarco`, we have `{query.doc, doc.query}` since there is no context.

For training, we choose [`ratio`](`ratio`) of dataset for training and create `{ctx.query.doc, ctx.doc.query}.train.tsv` file(s). 

Since our main purpose is to evaluate the retrieval power of refinements to the queries, we can input either of the following options w/ or w/o context and consider whaterver the model generates as a refinement to the query:

1. `ctx.query.*/pred.{refinement index}-{model checkpoint}`: query
2. `ctx.doc.*/pred.{refinement index}-{model checkpoint}`: relevant passages of a query 
3. `ctx.querydoc.*/pred.{refinement index}-{model checkpoint}`: concatenation of query and its relevant passages

We save the test file(s) as `{ctx.query.*, ctx.doc.*, ctx.querydoc.*}.test.tsv`.

## Run
Our run depends on T5 to generate the refinements to the input original queries. We can run T5 on local machine (cpu/gpu), or on google cloud (tpu), which is T5 pereferance,
1. [Local Machine (cpu/gpu)(Linux, Windows)](https://github.com/fani-lab/personalized_query_refinement/blob/main/RUNT5.md#localhost-cpu-or-gpu)
2. [Google Cloud (tpu)](https://github.com/fani-lab/personalized_query_refinement/blob/main/RUNT5.md#google-cloud-tpu)

## Results
We calculate the retrieval power of each query refinement on both train and test sets using IR metrics like `map` or `ndcg` compared to the original query and see if the refinements are better.

### MSMarco

### AOL
-- With User ID as context
-- Without UserID 

### Yandex

