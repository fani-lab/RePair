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


@REM --gin_param="t5.data.utils.f = './output/t5/vocabs/cc_all.32000/sentencepiece.model'" 
@REM --gin_file="./src/mdl/t5/t5/models/gin/dataset.gin" ^	
@REM --gin_param="SentencePieceVocabulary.sentencepiece_model_file = './output/t5/vocabs/cc_all.32000/sentencepiece.model'"  ^
@REM --gin_param="tsv_dataset_fn.vocabulary = @SentencePieceVocabulary()" ^ ==> Does not work since seqio does not import gin and this functiontion does not have gin.configurable decorator
@REM --gin_param="run.vocabulary = @SentencePieceVocabulary()" ==> even if works, at t5.models.mesh_transformer_main.py#181, it is explicitly bind to a function. So, useless gin_param
@REM The only way is to hard code vocab path 
@REM --module_import="numpy" ^ just to bypass the default "t5.data.mixtures" at t5.models.mesh_transformer_main#57 that cause the get_default_vocabulary() to be called very soon!