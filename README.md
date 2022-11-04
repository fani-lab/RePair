# Personalized Query Refinement

## About this project

Creating alternative queries, also known as query reformulation, has been shown to improve users' search experience.  Recently, neural seq-to-seq-based models have shown increased effectiveness when applied for learning to reformulate an original query into effective alternate queries. Such methods however forgo incorporating user information. In this paper, we investigate personalized query reformulation by incorporating user information on the performance of neural query reformulation methods compared to the lack thereof on AOL and Yandex query search logs. Our experiments demonstrate the synergistic effects of taking user information into account for query reformulation.


## Installation
You need to have ``Python=3.8`` and install [`pyserini`](https://github.com/castorini/pyserini/) package, among others listed in [``requirements.txt``](requirements.txt):

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

### Steps to run the T5 Training:

We adopt the [docTTTTTTQuery](https://github.com/castorini/docTTTTTquery#learning-a-new-prediction-model-t5-training-with-tensorflow) training methodology to fine-tune our T5 Model on 3 datasets. 
\
\
Our code creates the Doc-query pairs in the __Data__ folder for every dataset from where we push the code into a storage bucket in GCS.

We first train the T5 and infer the results by generating __n__ alternative queries (N=10) using Google Cloud TPU Storage and tensorflow.
We measure various metrics from the alternative queries and keep only the queries that have a higher metric than the existing query

## Results


### MSMarco
### AOL

-- With User ID as context
\
\
-- Without UserID 

### Yandex


**Note:** when installing jdk11, remember to check your env path for the executable in windows OS.

