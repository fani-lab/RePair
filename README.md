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
We create training sets based on different pairings of queries and relevant passages in the `./data/preprocessed/{domain name}/` for each domain like [`./data/preprocessed/msmarco/`](./data/preprocessed/msmarco/) for `msmarco`.

1. `ctx.query.doc`: context: query -> relevant passages
2. `ctx.doc.query`: context: relevant passages -> queries like in [docTTTTTTQuery](https://github.com/castorini/docTTTTTquery#learning-a-new-prediction-model-t5-training-with-tensorflow)

where the context will be `userid` (personalized) or `none` (blind). 

For training, we choose [`ratio`](`ratio`) of dataset for training and create `{ctx.query.doc, ctx.doc.query}.train.tsv` file(s). 

Since our main purpose is to evaluate the retrieval power of refinements to the queries, we can input either of the following options w/ or w/o context and consider whaterver the model generates as a refinement to the query:

1. `ctx.query`: query
2. `ctx.doc`: relevant passages of a query 
3. `ctx.query.doc`: concatenation of query and its relevant passages

We save the test file(s) as `{ctx.query, ctx.doc, ctx.query.doc}.test.tsv`.

#### Localhost (GPU)
**Windows:** T5 uses []() as default SentencePieceModel (word vocabularly) at google cloud for creating tasks. Also, the pretrained models are stored in google cloud. However, 
1. Google cloud file system protocol (`gs://`) is not supported by Windows yet (see [`here`](https://github.com/tensorflow/tensorflow/issues/38477)). Therefore, we need to download the pretrained models and SenetencePieceModel to our local machine. To do so, we need to install [`gsutil`](https://cloud.google.com/storage/docs/gsutil)
2. The google clound path for the default SentencePieceModel is hard coded at [`t5.data.DEFAULT_SPM_PATH`](https://github.com/google-research/text-to-text-transfer-transformer/blob/1b8375efe41f208f7f5c0744d57d7d913fa1eac4/t5/data/utils.py#L20). Therefore, it cannot be adjusted by `gin_params` or `gin config files` when calling T5's binary api (Mesh TensorFlow Transformer) as raised [`here`](https://github.com/google-research/text-to-text-transfer-transformer/issues/513)

- The solution is either explicitly change the path to a local model file like 
```
#https://github.com/google-research/text-to-text-transfer-transformer/t5/data/utils.py#L20
#DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS
DEFAULT_SPM_PATH = './output/t5/vocabs/cc_all.32000/sentencepiece.model'  # GCS
```
- Or prgramatically run T5 as in [here](https://github.com/google-research/text-to-text-transfer-transformer/tree/main/notebooks), 
- Or use our [`workaround`](https://github.com/fani-lab/text-to-text-transfer-transformer/blob/a9bb744d3e9811e912fddd7bfecf4d5334d00090/t5/data/utils.py#L24) to expose it as gin_param and call T5 as below:

```
SET PRETRAINED_STEPS=1000000
SET FINETUNE_STEPS=100
SET /a STEPS = %PRETRAINED_STEPS% + %FINETUNE_STEPS%

t5_mesh_transformer ^
--module_import="numpy" ^
--model_dir=".\\output\\t5_" ^
--gin_file="./output/t5/small/operative_config.gin" ^
--gin_param="utils.run.mesh_shape = 'model:1,batch:1'" ^
--gin_param="utils.run.mesh_devices = ['gpu:0']" ^
--gin_param="run.train_steps = %STEPS%" ^
--gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.tsv_dataset_fn" ^
--gin_param="tsv_dataset_fn.filename = './data/preprocessed/toy.msmarco/ctx.doc.query.train.tsv'" ^
--gin_param="utils.run.init_checkpoint = './output/t5/small/model.ckpt-1000000'" ^
--gin_param="MIXTURE_NAME = ''" ^
--gin_param="get_default_spm_path.path = './output/t5/vocabs/cc_all.32000/sentencepiece.model'"
```

Note that in Windows, `\\` should be used for `--model_dir` flag. Also, if the pre-trained model has already been trained for `n` steps and we need to fine-tune for another `m` steps, we have to pass `--gin_param="run.train_steps = {n+m}"`

**Unix-based:**

#### Google Cloud (TPU)
https://cloud.google.com/storage/docs/gsutil

https://ai.google.com/research/NaturalQuestions

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

Alternatively, we can navigate to [`console.cloud.google.com`](https://console.cloud.google.com/) >> search for `create a cloud tpu` from the search bar >> choose the zone as `us-central1-a` (this is where we can get accelerator v3-8) >> choose the TPU VM architecture and TPU type as `v3-8` >> choose TPU software version as `tpu-vm-tf-2.10.0` >> under the management section choose the `preemptibility` option (this will cost you much lower and you will only have the TPU running for 24 hours.) 

Now we can `ssh` to our virtual TPU to install `T5` and train it:

```sh
gcloud compute tpus tpu-vm ssh tpu-name --zone us-central1-a
pip install t5[gcp]
```

_Note: We need to disconnect the shell and connect again so the terminal refresh the environment and T5 become available._

**To train (fine-tuned) the model:**

```sh
t5_mesh_transformer  \
  --tpu='local' \
  --gcp_project="{project_id}" \
  --tpu_zone="us-central1-a" \
  --model_dir="gs://{bucket_name}/models/" \
  --gin_param="init_checkpoint = 'gs://t5-data/pretrained_models/base/model.ckpt-999900'" \
  --gin_file="dataset.gin" \
  --gin_file="models/bi_v1.gin" \
  --gin_file="gs://t5-data/pretrained_models/base/operative_config.gin" \
  --gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.tsv_dataset_fn" \
  --gin_param="tsv_dataset_fn.filename = 'gs://{bucket_name}/data/{ctx.query.doc, ctx.doc.query}.train.tsv'" \
  --gin_file="learning_rate_schedules/constant_0_001.gin" \
  --gin_param="run.train_steps = 1004000" \
  --gin_param="tokens_per_batch = 131072" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology ='v3-8'"
```
You should change `project_id`, and `bucket_name` accordingly. This will fine-tune the pretrained `gs://t5-data/pretrained_models/base/model.ckpt-999900` model on `gs://{bucket_name}/data/{ctx.query.doc or ctx.doc.query}.train.tsv` and save its checkpoints in the `gs://{bucket_name}/models/` folder of our storage bucket. 

**To produce the query refinements** (equivalently, to test the model), we ask the trained T5 to generate `n` outputs for each instance in the test file `gs://{bucket_name}/data/{ctx.query, ctx.doc, ctx.query.doc}.test.tsv` and save them in `gs://{bucket_name}/data/{ctx.query, ctx.doc, ctx.query.doc}.test.pred.tsv`:
```
t5_mesh_transformer \
  --tpu="local" \
  --gcp_project="{project_id}" \
  --tpu_zone="us-central1-a" \
  --model_dir="gs://{bucket_name}/models/" \
  --gin_file="gs://t5-data/pretrained_models/base/operative_config.gin" \
  --gin_file="infer.gin" \
  --gin_file="sample_decode.gin" \
  --gin_param="infer_checkpoint_step = 1004000" \
  --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 64}" \
  --gin_param="Bitransformer.decode.max_decode_length = 64" \
  --gin_param="input_filename = 'gs://{bucket_name}/data/{ctx.query, ctx.doc, ctx.query.doc}.test.tsv'" \
  --gin_param="output_filename = 'gs://{bucket_name}/data/{ctx.query, ctx.doc, ctx.query.doc}.test.pred.tsv'" \
  --gin_param="tokens_per_batch = 131072" \
  --gin_param="Bitransformer.decode.temperature = 1.0" \
  --gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = {n}"
```

## Results


Then, we calculate the retrieval power of each query refinement on both train and test sets using IR metrics like `map` or `ndcg` compared to the original query and see if the refinements are better.

### MSMarco

### AOL
-- With User ID as context
-- Without UserID 

### Yandex

