import functools, os
import tensorflow.compat.v1 as tf
# import tensorflow astfds

import t5
import t5.models
import t5.data.mixtures
import seqio

tf.disable_v2_behavior()
from contextlib import contextmanager


@contextmanager
def tf_verbosity_level(level):
  og_level = tf.logging.get_verbosity()
  tf.logging.set_verbosity(level)
  yield
  tf.logging.set_verbosity(og_level)


def prep(ds, task_name, in_type, out_type):
    def to_inputs_and_targets(line, task_name, in_type, out_type):
        """Map {"query": ..., "doc": ...}->{"inputs": ..., "targets": ...}."""
        return {
            "inputs": tf.strings.join([f'{task_name} question: ', line[in_type]]),
            "targets": line[out_type]
        }
    return ds.map(functools.partial(to_inputs_and_targets, task_name=task_name, in_type=in_type, out_type=out_type), num_parallel_calls=tf.data.experimental.AUTOTUNE)


#in_type or out_type \in {ctx}.{query, doc, docs}
def def_task(tsv_path, task_name, nexamples, in_type, out_type, vocab_model_path):
    def_out_feat = {
        "inputs": seqio.Feature(vocabulary=seqio.SentencePieceVocabulary(vocab_model_path, t5.data.DEFAULT_EXTRA_IDS), add_eos=True),
        "targets": seqio.Feature(vocabulary=seqio.SentencePieceVocabulary(vocab_model_path, t5.data.DEFAULT_EXTRA_IDS), add_eos=True)
    }

    seqio.TaskRegistry.add(
        task_name,
        source=seqio.TextLineDataSource(split_to_filepattern=tsv_path, num_input_examples=nexamples),
        preprocessors=[functools.partial(t5.data.preprocessors.parse_tsv, field_names=[in_type, out_type]),
                       functools.partial(prep, task_name=task_name, in_type=in_type, out_type=out_type),
                       seqio.preprocessors.tokenize_and_append_eos],
        postprocess_fn=t5.data.postprocessors.lower_text,
        metric_fns=[t5.evaluation.metrics.accuracy],
        output_features=def_out_feat,
    )

    # task = seqio.TaskRegistry.get(task_name)
    # ds = task.get_dataset(split="train", sequence_length={"inputs": 32, "targets": 256})
    # print("A few preprocessed train examples...")
    # for ex in tfds.as_numpy(ds.take(5)): print(ex)
    # ds = task.get_dataset(split="test", sequence_length={"inputs": 32, "targets": 256})
    # print("A few preprocessed validation examples...")
    # for ex in tfds.as_numpy(ds.take(5)): print(ex)


def finetune(tsv_path, pretrained_dir, steps, output, lseq, task_name, nexamples=None, in_type='query', out_type='doc', vocab_model_path='./../output/t5-data/vocabs/cc_en.32000/sentencepiece.model', gcloud=False):

    def_task(tsv_path, task_name, nexamples, in_type, out_type, vocab_model_path)
    # Limit number of checkpoints to fit within 5GB (if possible).
    model_parallelism, train_batch_size, keep_checkpoint_max = {"small": (1, 256, 16), "base": (2, 128, 8), "large": (8, 64, 4), "3B": (8, 16, 1), "11B": (8, 16, 1)}[os.path.split(pretrained_dir)[-1]]

    if not os.path.isdir(output): os.makedirs(output)
    if gcloud: import gcloud
    # Mesh Tensorflow Transformer
    model = t5.models.MtfModel(
        model_dir=output.replace('/', os.path.sep),
        tpu=gcloud.TPU_ADDRESS if gcloud else None,
        tpu_topology=gcloud.TPU_TOPOLOGY if gcloud else None,
        model_parallelism=model_parallelism,
        batch_size=train_batch_size,
        sequence_length=lseq,
        learning_rate_schedule=0.003,
        save_checkpoints_steps=100,#5000
        keep_checkpoint_max=keep_checkpoint_max if gcloud else None,
        iterations_per_loop=100,
    )

    model.finetune(
        mixture_or_task_name=task_name,
        pretrained_model_dir=pretrained_dir,
        finetune_steps=steps
    )

    # I think we don't need to export the model. It's saved by checkpoints
    # model.batch_size = 1  # make one prediction per call
    # saved_model_path = model.export(
    #     export_dir,
    #     checkpoint_step=-1,  # use most recent
    #     beam_size=1,  # no beam search
    #     temperature=1.0,  # sample according to predicted distribution
    # )
    # TODO: we can export the model
    return model


def predict(t5_model, model_dir, iter, query_file, output, lseq, model_name, vocab_model_path='./../output/t5-data/vocabs/cc_en.32000/sentencepiece.model', gcloud=False):

    if gcloud: import gcloud
    model_parallelism, train_batch_size, keep_checkpoint_max = {"small": (1, 256, 16), "base": (2, 128, 8), "large": (8, 64, 4), "3B": (8, 16, 1), "11B": (8, 16, 1)}[t5_model.split('.')[0]]
    model = t5.models.MtfModel(
        model_dir=model_dir,             #    f'{output}/model'.replace('/', os.path.sep),
        tpu=gcloud.TPU_ADDRESS if gcloud else None,
        tpu_topology=gcloud.TPU_TOPOLOGY if gcloud else None,
        model_parallelism=model_parallelism,
        batch_size=train_batch_size * 4,
        sequence_length=lseq,
    )

    with tf_verbosity_level('ERROR'):#There might be empty '' predictions!
        for i in range(iter):
            print(f'Predicting {str(i)}...')
            model.predict(
                input_file=query_file,
                output_file=f'{output}/{model_name}.{str(i)}'.replace('/', os.path.sep),
                checkpoint_steps=-1,#the last one
                beam_size=1, #int, a number >= 1 specifying the number of beams to use for
                temperature=1.0, #float, a value between 0 and 1 (must be 0 if beam_size > 1) 0.0 means argmax/most probable, 1.0 means sample according to predicted distribution.
                keep_top_k=-1,#integer, a value between 1 and the vocabulary size. When sampling, only pick tokens that are in the k most likely.
                vocabulary=seqio.SentencePieceVocabulary(vocab_model_path, t5.data.DEFAULT_EXTRA_IDS))


    # # since we do eval using IR, we don't need this.
    # try: seqio.TaskRegistry.get(task_name)
    # except ValueError as ve: def_task(tsv_path, task_name, nexamples, in_type, out_type, vocab_model_path)
    # model.eval(mixture_or_task_name=task_name,checkpoint_steps=-1)#"latest", could be "all"
    # ds = seqio.TaskRegistry.get(task_name).get_dataset(split=split, sequence_length=lseq, shuffle=False)
    # ## if all checkpoints
    # pred_files = tf.io.gfile.glob(os.path.join(output, f'{split}_eval/{task_name}_*_predictions'))
    # latest_pred_file = sorted(pred_files, key=lambda x: int(x.split("_")[-2]))[-1] #get most recent prediction file by sorting by their step.
    #
    # #collect (inputs, targets, prediction) from the dataset and predictions file
    # import random
    # results = []
    # with tf.io.gfile.GFile(latest_pred_file) as preds:
    #     for ex, pred in zip(tfds.as_numpy(ds), preds): results.append((tf.compat.as_text(ex["inputs_pretokenized"]), tf.compat.as_text(ex["targets_pretokenized"]),pred.strip()))
    #     print(f'Random predictions for {task_name} using checkpoint {latest_pred_file}')
    #     for inp, tgt, pred in random.choices(results, k=10): print(f'Input:{inp}\nTarget:{tgt}\nPrediction:{pred}\n')
