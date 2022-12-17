SET CUDA_VISIBLE_DEVICES=-1
@REM needs to use the fork verison of t5 for t5.data.utils.py
@ECHO OFF
@SET MODEL_DIR="..\\output\\t5.small.bat.query.doc"
@SET PRETRAINED_STEPS=1000000
@SET FINETUNE_STEPS=5
@SET /a STEPS = %PRETRAINED_STEPS% + %FINETUNE_STEPS%
@SET IN_LENGTH=32
@SET OUT_LENGTH=256
@SET GPU=0

t5_mesh_transformer ^
--module_import="numpy" ^
--model_dir=%MODEL_DIR% ^
--gin_file="../output/t5-data/pretrained_models/small/operative_config.gin" ^
--gin_param="utils.run.mesh_shape = 'model:1,batch:1'" ^
--gin_param="utils.run.mesh_devices = ['gpu:%GPU%']" ^
--gin_param="run.train_steps = %STEPS%" ^
--gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.tsv_dataset_fn" ^
--gin_param="utils.run.sequence_length = {'inputs': %IN_LENGTH%, 'targets': %OUT_LENGTH%}" ^
--gin_param="tsv_dataset_fn.filename = '../data/preprocessed/toy.msmarco/query.doc.train.tsv'" ^
--gin_param="utils.run.init_checkpoint = '../output/t5-data/pretrained_models/small/model.ckpt-1000000'" ^
--gin_param="MIXTURE_NAME = ''" ^
--gin_param="get_default_spm_path.path = '../output/t5-data/vocabs/cc_all.32000/sentencepiece.model'"

@REM TODO: save model: --export_dir "directory to export SavedModels to. Will use `model_dir` if unspecified."

FOR /l %%x IN (1, 1, 5) DO t5_mesh_transformer ^
--module_import="numpy" ^
--model_dir=%MODEL_DIR% ^
--gin_file="../output/t5-data/pretrained_models/small/operative_config.gin" ^
--gin_file="infer.gin" ^
--gin_file="sample_decode.gin" ^
--gin_param="utils.run.mesh_shape = 'model:1,batch:1'" ^
--gin_param="utils.run.mesh_devices = ['gpu:%GPU%']" ^
--gin_param="get_default_spm_path.path = '../output/t5-data/vocabs/cc_all.32000/sentencepiece.model'" ^
--gin_param="utils.run.mode = 'infer'" ^
--gin_param="utils.run.eval_checkpoint_step = -1" ^
--gin_param="utils.run.sequence_length = {'inputs': %IN_LENGTH%, 'targets': %OUT_LENGTH%}" ^
--gin_param="input_filename = '../data/preprocessed/toy.msmarco/query.doc.test.tsv'" ^
--gin_param="output_filename = '..\\output\\t5.small.bat.query.doc\\pred.$ITER'" ^
--gin_param="Bitransformer.decode.temperature = 1.0" ^
--gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = -1" ^
--gin_param="Bitransformer.decode.beam_size = 1" 
