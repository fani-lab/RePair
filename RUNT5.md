## Localhost (cpu or gpu)

### Unix-based

### Windows
T5's pretrained models are stored in [`google cloud`](https://console.cloud.google.com/storage/browser/t5-data/pretrained_models/). Also, the default word vocabularly is defined as a SentencePieceModel based on [`C4`](https://console.cloud.google.com/storage/browser/t5-data/vocabs/cc_all.32000) corpus. However, 
1. Windows does not support google cloud file system protocol (`gs://`) yet (see [`here`](https://github.com/tensorflow/tensorflow/issues/38477)). We need to download the pretrained models and SenetencePieceModel(s) to our local machine. To do so, we need to install [`gsutil`](https://cloud.google.com/storage/docs/gsutil):

```
gsutil -m cp -r "gs://t5-data/pretrained_models/small" . # or base, large, 3B, 11B
gsutil -m cp -r "gs://t5-data/vocabs/cc_all.32000" "gs://t5-data/vocabs/cc_en.32000" .
```

2. The google cloud path for the default SentencePieceModel is hard coded at [`t5.data.DEFAULT_SPM_PATH`](https://github.com/google-research/text-to-text-transfer-transformer/blob/1b8375efe41f208f7f5c0744d57d7d913fa1eac4/t5/data/utils.py#L20)! Therefore, it cannot be adjusted by `gin_params` or `gin config files` when calling T5's binary api (Mesh TensorFlow Transformer) using local SentencePieceModel as raised [`here`](https://github.com/google-research/text-to-text-transfer-transformer/issues/513)

The solution is either: 
- Prgramatically run T5 as in our pipeline adopting [`t5-trivia`](https://github.com/google-research/text-to-text-transfer-transformer/blob/main/notebooks/t5-trivia.ipynb), 
- Or use our T5's fork [`here`](https://github.com/fani-lab/text-to-text-transfer-transformer/blob/a9bb744d3e9811e912fddd7bfecf4d5334d00090/t5/data/utils.py#L24) to expose it as gin_param and call T5 for train and test in a batch file like [`./src/mt5.bat`](./src/mt5.bat). Note that T5 uses `os.path.sep` to modify file paths. So, in Windows, `\\` should be used for `--model_dir` flag. Also, if the pre-trained model has already been trained for `n` steps and we need to fine-tune for another `m` steps, we have to pass `--gin_param="run.train_steps = {n+m}"`. For instance, our batch file [`./src/mt5.bat`](./src/mt5.bat) locally fine-tune the T5-small's model that are pretrained on C4 `"utils.run.init_checkpoint = './output/t5/small/model.ckpt-1000000'"` for another `@SET FINETUNE_STEPS=5` steps on our training set of `"tsv_dataset_fn.filename = './data/preprocessed/toy.msmarco/doc.query.train.tsv'"` using the local SentencePieceModel at `"get_default_spm_path.path = './output/t5/vocabs/cc_all.32000/sentencepiece.model'`

- Or explicitly change the path in T5 installed package to a local model file like 
```
#https://github.com/google-research/text-to-text-transfer-transformer/t5/data/utils.py#L20
#DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS
DEFAULT_SPM_PATH = './output/t5/vocabs/cc_all.32000/sentencepiece.model'  # Local
```

## Google Cloud (tpu)
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
