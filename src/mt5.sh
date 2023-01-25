pip install t5[gcp]
#TPU will be always local
TRAINSTEPS=1004000
GCP_PROJECT="replace with YOUR PROJECT ID"
TPU_ZONE="replace with YOUR TPU ZONE"
YOUR_BUCKET="replace with your TPU bucket"
PAIRING="docs.query"
N=25

#mesh transformer fine tuning. Replace run steps to define the number of iterations to run on T5 on. 
t5_mesh_transformer  \
  --tpu='local' \
  --gcp_project=$GCP_PROJECT \
  --tpu_zone=$TPU_ZONE \
  --model_dir="gs://$YOUR_BUCKET/models/$DSQPAIRING" \
  --gin_param="init_checkpoint = 'gs://t5-data/pretrained_models/base/model.ckpt-999900'" \
  --gin_file="dataset.gin" \
  --gin_file="models/bi_v1.gin" \
  --gin_file="gs://t5-data/pretrained_models/base/operative_config.gin" \
  --gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.tsv_dataset_fn" \
  --gin_param="tsv_dataset_fn.filename = 'gs://$YOUR_BUCKET/data/$DSQPAIRING/$DSQPAIRING.train.tsv'" \
  --gin_file="learning_rate_schedules/constant_0_001.gin" \
  --gin_param="run.train_steps = $TRAINSTEPS" \
  --gin_param="tokens_per_batch = 131072" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology ='v3-8'"


#uses the fine tuned model to predict. Replace inputs and target values
# query doc pairing has inputs:64 targets: 512
# doc query pairing has inputs:512 targets: 64
#max decode length should be the same as targets
for ITER in {1..25}; do 
t5_mesh_transformer \
  --tpu='local' \
  --gcp_project=$GCP_PROJECT \
  --tpu_zone=$TPU_ZONE \
  --model_dir="gs://$YOUR_BUCKET/models/$DSQPAIRING" \
  --gin_file="gs://t5-data/pretrained_models/base/operative_config.gin" \
  --gin_file="infer.gin" \
  --gin_file="sample_decode.gin" \
  --gin_param="infer_checkpoint_step = $TRAINSTEPS" \
  --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 64}" \
  --gin_param="Bitransformer.decode.max_decode_length = 64" \
  --gin_param="input_filename = 'gs://$YOUR_BUCKET/data/$DSQPAIRING/$DSQPAIRING.test.tsv'" \
  --gin_param="output_filename = 'gs://$YOUR_BUCKET/data/$DSQPAIRING/pred$ITER.tsv'" \
  --gin_param="tokens_per_batch = 131072" \
  --gin_param="Bitransformer.decode.temperature = 1.0" \
  --gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = 10" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology ='v3-8'"
done
