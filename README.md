# ``No Query Left Behind``: Query Refinement via Backtranslation
Web users often struggle to express their information needs clearly in short, vague queries, making it hard for search engines to find relevant results. Query refinement, which aims to improve search relevance by adjusting original queries, is crucial in addressing this challenge. However, current evaluation methods for query refinement models may not accurately reflect real-world usage patterns. We propose a novel approach using natural language backtranslation to create benchmark datasets for evaluating query refinement models. Backtranslation involves translating a query from one language to another and then translating it back, ensuring that the meaning remains consistent. We believe that backtranslation can:

1. Identify missing terms in a query that are assumed to be understood due to their common usage in the original language.
2. Include relevant synonyms from the target language to provide additional context.
3. Clarify the meaning of ambiguous terms or phrases.

We conducted extensive experiments using widely recognized TREC query sets and multiple languages. Our results, evaluated using various metrics, demonstrate the effectiveness of backtranslation in creating high-quality benchmark datasets for evaluating query refinement methods.

<table align="center" border=0>
<tr>
<td >

- [1. Setup](#1-setup)
  * [Lucene Indexes](#lucene-indexes)
- [2. Quickstart](#2-quickstart)
  * [`query_refinement`](#query_refinement)
  * [`similarity`](#similarity)
  * [`search`](#search)
  * [`eval`](#eval)
  * [`agg, box`](#agg-box)
- [3. Gold Standard Datasets](#3-gold-standard-datasets)
  * [File Structure](#file-structure)
  * [Settings](#settings)
  * [Stats](#stats)
  * [Samples](#samples)
- [4. Acknowledgement](#4-acknowledgement)
- [5. License](#5-license)
- [6. Citation](#6-citation)

</table>

![image](misc/flow.png)

## 1. Setup
You need to have ``Python=3.8`` and install [`pyserini`](https://github.com/castorini/pyserini/) package (needs `Java`), among others listed in [``requirements.txt``](requirements.txt). 
You may also need to install [anserini](https://github.com/castorini/anserini). Only for indexing purposes and RelevanceFeedback refiner.
> [!IMPORTANT]   
> Anserini is only compatible with Java version 11. Using versions older or newer than this will result in an error.
>
We also suggest you clone our repo with the `--recurse-submodules` (alternatively, use the `git submodule update --init` inside the cloned repository) to get the [`trec_eval`](https://github.com/usnistgov/trec_eval) metric evaluation tool:

By ``pip``, clone the codebase and install the required packages:
```sh
pip install -r requirements.txt
```

By [``conda``](https://www.anaconda.com/products/individual):

```sh
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
To perform fast IR tasks, we need to build the sparse indexes of document corpora or use the [`prebuilt-indexes`](https://github.com/castorini/pyserini/blob/master/docs/prebuilt-indexes.md). The path to the index need to be set in [`./src/param.py`](./src/param.py).

## 2. Quickstart
For using query refinement make sure to add the command to the pipeline in the [./src/param.py](./src/param.py).

As seen in the above [`workflow`](./misc/workflow.png), `RePair` has three pipelined steps: 
> 1. Refining Quereis: [`query_refinement`]
> 2. Performance Evaluation: [`search`, `eval`]
> 3. Dataset Curation: [`agg`, `box`]

To run `RePair` pipeline, we need to set the required parameters of each step in [`./src/param.py`](./src/param.py) such as the ranker used in the search step. Then, the pipeline can be run by its driver at [`./src/main.py`](./src/main.py):

```sh
python -u main.py -data ../data/raw/toy.msmarco.passage -domain msmarco.passage
```
```sh
python -u main.py -data ../data/raw/toy.aol-ia -domain aol-ia
```
```sh
python -u main.py -data ../data/raw/toy.msmarco.passage ../data/raw/toy.aol-ia -domain msmarco.passage aol-ia
```
### [`['query_refinement']`](./src/refinement/refiner_param.py#L9)

# Refiners
The objective of query refinement is to produce a set of potential candidate queries that can function as enhanced and improved versions. This involves systematically applying various unsupervised query refinement techniques to each query within the input dataset.

<table align="center" border=0>
<thead>
  <tr><td colspan="3" style="background-color: white;"><img src="./misc/classdiagram.png", width="1000", alt="ReQue: Class Diagram"></td></tr>     
  <tr><td colspan="3">
      <p align="center">Class Diagram for Query Refiners in <a href="./src/refinement/">src/refinement/</a>. [<a href="https://app.lucidchart.com/documents/view/64fedbb0-b385-4696-9adc-b89bc06e84ba/HWEp-vi-RSFO">zoom in!</a>].</p>
      <p align="center"> The expanders are initialized by the Expander Factory in <a href="./src/refinement/refiner_factory.py">src/refinement/refiner_factory.py</a></p></td></tr> 
 </thead>
</table>

Here is the list of queries:
| **Expander** 	| **Category** 	| **Analyze type** 	|
|---	|:---:	|:---:	|
| adaponfields 	| Top_Documents 	| Local 	|
| anchor 	| Anchor_Text 	| Global 	|
| [backtranslation](#Backtranslation) 	| Machine_Translation 	| Global 	|
| bertqe] 	| Top_Documents 	| Local 	|
| conceptluster 	| Concept_Clustering 	| Local 	|
| conceptnet 	| Semantic_Analysis 	| Global 	|
| docluster 	| Document_Summaries 	| Local 	|
| glove 	| Semantic_Analysis 	| Global 	|
| onfields 	| Top_Documents 	| Local 	|
| relevancefeedback 	| Top_Documents 	| Local 	|
| rm3 	| Top_Documents 	| Local 	|
| sensedisambiguation 	| Semantic_Analysis 	| Global 	|
| stem.krovetz 	| Stemming_Analysis 	| Global 	|
| stem.lovins 	| Stemming_Analysis 	| Global 	|
| stem.paicehusk 	| Stemming_Analysis 	| Global 	|
| stem.porter 	| Stemming_Analysis 	| Global 	|
| stem.sstemmer 	| Stemming_Analysis 	| Global 	|
| stem.trunc 	| Stemming_Analysis 	| Global 	|
| tagmee 	| Wikipedia 	| Global 	|
| termluster 	| Term_Clustering 	| Local 	|
| thesaurus 	| Semantic_Analysis 	| Global 	|
| wiki 	| Wikipedia 	| Global 	|
| word2vec 	| Semantic_Analysis 	| Global 	|
| wordnet 	| Semantic_Analysis 	| Global 	|

# Backtranslation
Back translation, also known as reverse translation or dual translation, involves translating content, whether it is a query or paragraph, from one language to another and retranslating it to the original language. This method provides several options for the owner to make a decision that makes the most sense based on the task at hand.
For additional details, please refer to this [document](./misc/Backtranslation.pdf).

## Example
| **q** 	| **map q** 	| **language** 	| **translated q** 	| **backtranslated q** 	| **map q'** 	|
|---	|:---:	|:---:	|:---:	|:---:	|:---:	|
| Italian nobel prize winners 	| 0.2282 	| farsi 	| برندهای جایزه نوبل ایتالیایی 	| Italian Nobel laureates 	| 0.5665 	|
| banana paper making 	| 0.1111 	| korean 	| 바나나 종이 제조 	| Manufacture of banana paper 	| 1 	|
| figs 	| 0.0419 	| tamil 	|  அத்திமரங்கள்  	| The fig trees 	| 0.0709 	|


### [`['similarity']`](./src/param.py#L12)

To evaluate the quality of the refined queries, metrics such as bleu, rouge, and semsim are employed. The bleu score measures the similarity between the backtranslated and original query by analyzing n-grams, while the rouge score considers the overlap of n-grams to capture essential content. Due to their simplicity and effectiveness, these metrics are widely utilized in machine translation tasks. Despite their usefulness, both scores may not accurately capture the overall meaning or fluency of the translated text due to their heavy reliance on n-grams. To address topic drift and evaluate the similarity between the original and refined queries, we additionally employ [declutr](https://aclanthology.org/2021.acl-long.72/) for query embeddings, computing cosine similarity. Declutr, a self-learning technique requiring no labeled data, minimizes the performance gap between unsupervised and supervised pretraining for universal sentence encoders during the extension of transformer-based language model training. The semsim metric, relying on cosine similarity of embeddings, proves highly effective in capturing the subtle semantic nuances of language, establishing itself as a dependable measure of the quality of backtranslated queries.

The below images demonstrate the average token count for the original queries in English and their backtranslated versions across various languages, along with the average pairwise semantic similarities measured using 'rouge' and 'declutr'. It's evident that all languages were able to introduce new terms into the backtranslated queries while maintaining semantic coherence.
![image](misc/similarity.png)

## Example
These samples are taken from an ANTIQUE dataset that has been refined using a backtranslation refiner with the German language.
| **id** 	| **original** 	| **refined** 	| **rouge1** 	| **rouge2** 	| **rougeL** 	| **rougeLsum** 	| **bleu** 	| **precisions** 	| **brevity_penalty** 	| **length_ratio** 	| **translation_length** 	| **reference_length** 	| **semsim** 	|
|---	|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|:---:	|
| 1290612 	| why do anxiety and depression seem to coexist? 	| Why do fear and depression seem to be linked 	| 0.705882 	| 0.533333 	| 0.705882 	| 0.705882353 	| 0.315598 	| [0.5555555555555556, 0.375, 0.2857142857142857,   0.16666666666666666] 	| 1 	| 1 	| 9 	| 9 	| 0.8554905 	|
| 4473331 	| How can I keep my   rabit indoors? 	| How can I keep my   rabbit in the house 	| 0.625 	| 0.571429 	| 0.625 	| 0.625 	| 0.446324 	| [0.5555555555555556,   0.5, 0.42857142857142855, 0.3333333333333333] 	| 1 	| 1.125 	| 9 	| 8 	| 0.7701595 	|
| 1509982 	| How is the Chemistry is a basic of Science? 	| How is chemistry a principle of science 	| 0.75 	| 0.285714 	| 0.75 	| 0.75 	| 0 	| [0.5714285714285714, 0.16666666666666666, 0.0, 0.0] 	| 0.651439058 	| 0.7 	| 7 	| 10 	| 0.7796929 	|


### [`['search']`](./src/param.py#L17)
We search the relevant documents for both the original query and each of the `potential` refined queries. We need to set an information retrieval method, called ranker, that retrieves relevant documents and ranks them based on relevance scores. We integrate [`pyserini`](https://github.com/castorini/pyserini), which provides efficient implementations of sparse and dense rankers, including `bm25` and `qld` (query likelihood with Dirichlet smoothing). 


### [`['eval']`](./src/param.py#L20)
The search results of each potential refined queries are evaluated based on how they improve the performance with respect to an evaluation metric like `map` or `mrr`. 


### [`['agg', 'box']`](./src/param.py#L12)
Finaly, we keep those potential refined queries whose performance (metric score) have been better or equal compared to the original query, i.e., `refined_query_metric >= original_query_metric and refined_q_metric > 0`.

We keep two main datasets as the outcome of the `RePair` pipeline:

> 1. `./output/{input query set}/{transformer name}.{pairing strategy}/{ranker}.{metric}.agg.gold.tsv`: contains the original queries and their refined queries that garanteed the `better` performance along with the performance metric values

> 2. `./output/{input query set}/{transformer name}.{pairing strategy}/{ranker}.{metric}.agg.all.tsv`: contains the original queries and `all` their predicted refined queries along with the performance metric values

For boxing, since we keep the performances for all the potential queries, we can change the condition and have a customized selection like having [`diamond`](https://dl.acm.org/doi/abs/10.1145/3459637.3482009) refined queries with maximum possible performance, i.e., `1` by setting the condition: `refined_query_metric >= original_query_metric and refined_q_metric == 1`. The boxing condition can be set at [`./src/param.py`](./src/param.py#L12). 

```
'box': {'gold':     'refined_q_metric >= original_q_metric and refined_q_metric > 0',
        'platinum': 'refined_q_metric > original_q_metric',
        'diamond':  'refined_q_metric > original_q_metric and refined_q_metric == 1'}
```

After this step, the final structure of the output will be look like below:

```bash
├── output
│   └── dataset_name
│       └── refined_queries_files
│   │   └── ranker.metric [such as bm25.map]
│   │       └── [This is where all the results from the search, eval, aggregate, and boxing are stored]

```


### Settings
We've created benchmark query refinement datasets for the 'trec' dataset using the 'backtranslated' refiner with both 'bm25' and 'qld' rankers, along with 'map' and 'qld' evaluation metrics.You can adjust the settings [./src/param.py](./src/param.py)

### Samples


## 4. Acknowledgement
We benefit from [``trec_eval``](https://github.com/usnistgov/trec_eval), [``pyserini``](https://github.com/castorini/pyserini), [``ir-dataset``](https://ir-datasets.com/), and other libraries. We would like to thank the authors of these libraries and helpful resources.
  
## 5. License
©2023. This work is licensed under a [CC BY-NC-SA 4.0](license.txt) license.



