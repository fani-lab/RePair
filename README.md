# Personalized Query Refinement
Creating alternative queries, also known as query reformulation, has been shown to improve users' search experience.  Recently, neural transformers like `(* * * )` have shown increased effectiveness when applied for learning to reformulate an original query into effective alternate queries. Such methods however forgo incorporating user information. We investigate personalized query reformulation by incorporating user information on the performance of neural query reformulation methods compared to the lack thereof on `aol` and `yandex` query search logs. Our experiments demonstrate the synergistic effects of taking user information into account for query reformulation.


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
_Note: When installing jdk11, remember to check your env path for the executable in Windows._

## Model Training and Test
We use [`T5`](https://github.com/google-research/text-to-text-transfer-transformer) to train a model, that when given an input query (origianl query), generates refined (better) versions of the query in terms of retrieving more relevant documents at higher ranking positions. We adopt the [`docTTTTTTQuery`](https://github.com/castorini/docTTTTTquery#learning-a-new-prediction-model-t5-training-with-tensorflow) training methodology to fine-tune `T5` model on `msmarco` (w/o `userid`) and `aol` (w/ and w/o `userid`). For `yandex` dataset, we need to train `T5` from scratch since the tokens are anonymized by random Ids. 

`check [T5X](https://github.com/google-research/t5x)`

### Dataset
We create the dataset based on different pairings of queries and relevant passages in the `./data/preprocessed/{domain name}/` for each domain like [`./data/preprocessed/msmarco/`](./data/preprocessed/msmarco/) for `msmarco`.

(1) `c:q>p`: context: query -> relevant passages

(2) `c:p>q`: context: relevant passages -> queries like in [docTTTTTTQuery](https://github.com/castorini/docTTTTTquery#learning-a-new-prediction-model-t5-training-with-tensorflow)

where the context will be `userid` (personalized) or `none` (blind). 

For training, we choose [`ratio`](`ratio`) of dataset for training. Since our main purpose is to evaluate the retrieval power of refinements to the queries, we can input either of the following w/ or w/o context and consider whaterver the model generates as a refinement to the query:

(a) `c:q`: query 

(b) `c:p`: relevant passages of a query 

(c) `c:qp`: concatenation of query and its relevant passages

#### Localhost

#### Google Cloud:
To proceed, we need a google cloud platform account and an active project. We need to push the dataset to cloud storage bucket created in the google cloud storage:

```sh
#create a bucket 
gcloud storage buckets create gs://{bucket_name}

# push the dataset 
gsutil cp {dataset} gs://{bucket_name}/data/
```

We need to create a TPU virtual machine by downloading the gcloud tool for the terminal and download it using

```sh
gcloud compute tpus tpu-vm create tpu-name --zone=us-central1-a --accelerator-type=v3-8 --version=tpu-vm-tf-2.10.0
```

Alternatively, we can navigate to [`cloud.google.com`](https://www.cloud.google.com) >> search for `create a cloud tpu` from the search bar >> choose the zone as `us-central1-a` (this is where we can get accelerator v3-8) >> choose the TPU VM architecture and TPU type as `v3-8` >> choose TPU software version as `tpu-vm-tf-2.10.0` >> under the management section choose the `preemptibility` option (this will cost you much lower and you will only have the TPU running for 24 hours.) 

Now we can `ssh` to our virtual TPU to install `T5` and train it:

```sh
gcloud compute tpus tpu-vm ssh tpu-name --zone us-central1-a
pip install t5[gcp]
```

_Note: Once the installation is done, we need to disconnect the shell and connect again so the terminal refresh the path into the bash shell._
`not sure I understood this`

To train (fine-tune) the model: 

```sh
t5_mesh_transformer  \
  --tpu='local' \
  --gcp_project="your_project_id" \
  --tpu_zone="your_tpu_zone" \
  --model_dir="gs://your_bucket/models/" \
  --gin_param="init_checkpoint = 'gs://t5-data/pretrained_models/base/model.ckpt-999900'" \
  --gin_file="dataset.gin" \
  --gin_file="models/bi_v1.gin" \
  --gin_file="gs://t5-data/pretrained_models/base/operative_config.gin" \
  --gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.tsv_dataset_fn" \
  --gin_param="tsv_dataset_fn.filename = 'gs://your_bucket/data/doc_query_pairs.train.tsv'" \
  --gin_file="learning_rate_schedules/constant_0_001.gin" \
  --gin_param="run.train_steps = 1004000" \
  --gin_param="tokens_per_batch = 131072" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology ='v3-8'"
```

This will train the model and save its checkpoints in the `gs://your_bucket/models/` folder of our storage bucket. 

## Results
To produce the query refinements, we ask the trained T5 to generate `n` outputs (N=10). 
`where we determine this?`

Then, we calculate the retrieval power of each query refinement on both train and test sets using IR metrics like `map` or `ndcg` compared to the original query and see if the refinements are better.

### MSMarco

### AOL
-- With User ID as context
-- Without UserID 

### Yandex

