#  RAW

Stores all the dataset from which we prepare data to be trained by a transformer model.



### msmarco.passage
The msmarco passage collection is readily available with qrels and queries. This can be found at: [msmarco](https://microsoft.github.io/msmarco/)
We make use of only two files.
1. qrels.train.tsv
2. queries.train.tsv

We provide the lucene index for msmarco or it can be readily obtained by [following these instructions to index using pyserini](https://github.com/castorini/pyserini/blob/master/docs/prebuilt-indexes.md)

### aol-ia
Unlike msmarco passage, aol-ia does not come with qrels, queries or index that can be readily obtained. We make it easy by providing all the files that can be used for other projects as a **resource**. 
1. qrels.tsv
2. queries.tsv
3. downloaded_docs

The index index will be automatically created with the downloaded docs by creating a jsonl collection and using that to create sparse retrieval index using pyserini.

**Note:** our next update will include dense retrieval index!


### Known issues and their solutions :
If you encounter any issues, please be sure to check out our issue page and the solutions 

- [AOL issues encountered while handling large files and processing large ranker docs into chunks for trec-eval](https://github.com/fani-lab/personalized_query_refinement/issues/11 )
- [AOL log download from internet archive](https://github.com/fani-lab/personalized_query_refinement/issues/7) :- **If you choose to downlaod the whole document yourself**
- [LZ4_decompress error on AOL. Happened because of corrupted files from web archive files.](https://github.com/fani-lab/personalized_query_refinement/issues/9) :- **we have provided the downloaded docs, which solves this issue.**
- [LuceneSearcher cannot find the index](https://github.com/fani-lab/RePair/issues/3)
