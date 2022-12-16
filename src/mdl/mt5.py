import functools, os, sys, time
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

import t5
import t5.models
import seqio

tf.disable_v2_behavior()
from contextlib import contextmanager
import logging as py_logging
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
