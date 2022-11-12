# Personalized Query Refinement

## About this project

Creating alternative queries, also known as query reformulation, has been shown to improve users' search experience.  Recently, neural seq-to-seq-based models have shown increased effectiveness when applied for learning to reformulate an original query into effective alternate queries. Such methods however forgo incorporating user information. We investigate personalized query reformulation by incorporating user information on the performance of neural query reformulation methods compared to the lack thereof on AOL and Yandex query search logs. Our experiments demonstrate the synergistic effects of taking user information into account for query reformulation.


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

### Installation / creating a storage bucket and TPU on Google cloud platform:

We adopt the [docTTTTTTQuery](https://github.com/castorini/docTTTTTquery#learning-a-new-prediction-model-t5-training-with-tensorflow) training methodology to fine-tune our T5 Model on msmarco, AOL and yandex dataset. 
\
\
Our code creates the Doc-query pairs,  in the __Data__/__preprocessed__/__dataset_name__/ folder for every dataset.

Move to the folder where the doc query pairs are created and push them to the data folder in your cloud storage bucket created in the Google Cloud storage. Follow these commands:

### creates a bucket 
```sh
gcloud storage buckets create gs://BUCKET_NAME
```

### Push the doc-query pair to the Data folder.

```sh
gsutil cp doc-query.train.tsv  gs://your_bucket/data/
```

This will push the file to the storage bucket folder called data. 


You need a google cloud platform account and an active project to proceed with this. 

\\
you can create a TPU by going to downloading the gcloud tool for the terminal and download it using

```sh
gcloud compute tpus tpu-vm create tpu-name \
--zone=us-central1-a \
--accelerator-type=v3-8 \
--version=tpu-vm-tf-2.10.0
```

or you can navigate to [cloud.google.com](https://www.cloud.google.com) search for **create a cloud tpu** from the search bar. 
 - Choose the zone as us-central1-a (Note: this is where we can get accelerator v3-8)
 - Choose the TPU VM architecture and TPU type as **v3-8**
 - Choose TPU software version as **tpu-vm-tf-2.10.0**
 - under the management section choose the preemptibility option. This will cost you much lower and you will only have the TPU running for 24 hours. 


after we create a TPU virtual machine, we will have to ssh into the cloud TPU to train the model. This can be done by:
```sh
gcloud compute tpus tpu-vm ssh tpu-name \
  --zone us-central1-a
```


once that is done, we can install the **t5[gcp]** package using pip on the tpu vm. 

```sh
pip install t5[gcp]
```

once the installation is completed successfully, you will have to disconnect the shell session and connect again for the terminal to refresh the path into the bash shell. 


### Training 

We first train the T5 and infer the results by generating __n__ alternative queries (N=10) using Google Cloud TPU Storage and tensorflow.
We measure various metrics from the alternative queries and keep only the queries that have a higher metric than the existing query

To train the model, we can follow this command 
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


This will train the model and store its checkpoint in the **/models** folder of our Storage bucket. 

## Results


### MSMarco
### AOL

-- With User ID as context
\
\
-- Without UserID 

### Yandex
\
\
**Note:** when installing jdk11, remember to check your env path for the executable in windows OS.

