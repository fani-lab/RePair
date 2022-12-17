@SET PRETRAINED_STEPS=1000000
@SET FINETUNE_STEPS=5
@SET /a STEPS = %PRETRAINED_STEPS% + %FINETUNE_STEPS%

t5_mesh_transformer ^
--module_import="numpy" ^
--model_dir="..\\output\\t5.small.bat.query.doc" ^
--gin_file="./output/t5-data/pretrained_models/small/operative_config.gin" ^
--gin_param="utils.run.mesh_shape = 'model:1,batch:1'" ^
--gin_param="utils.run.mesh_devices = ['gpu:0']" ^
--gin_param="run.train_steps = %STEPS%" ^
--gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.tsv_dataset_fn" ^
--gin_param="tsv_dataset_fn.filename = '../data/preprocessed/toy.msmarco/query.doc.train.tsv'" ^
--gin_param="utils.run.init_checkpoint = '../output/t5-data/pretrained_models/small/model.ckpt-1000000'" ^
--gin_param="MIXTURE_NAME = ''" ^
--gin_param="get_default_spm_path.path = '../output/t5-data/vocabs/cc_all.32000/sentencepiece.model'"


@REM TODO: save model: --export_dir "directory to export SavedModels to. Will use `model_dir` if unspecified."
@REM TODO: eval and/or predict
