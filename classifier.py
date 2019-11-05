
"""Train simple fastText-style classifier.

Inputs:
  words - text to classify
  ngrams - n char ngrams for each word in words
  labels - output classes to classify

Model:
  word embedding
  ngram embedding
  LogisticRegression classifier of embeddings to labels
"""
import inputs
import sys
import tensorflow as tf
import tensorflow.contrib as contrib
from tensorflow.contrib.layers import feature_column
from tensorflow.contrib.learn.python.learn.estimators.run_config import RunConfig

tf.flags.DEFINE_string("train_records", "data/tfrecords/train.tfrecords-*",
                       "Training file pattern for TFRecords, can use wildcards")
tf.flags.DEFINE_string("eval_records", "data/tfrecords/test.tfrecords-*",
                       "Evaluation file pattern for TFRecords, can use wildcards")
tf.flags.DEFINE_string("predict_records", None,
                       "File pattern for TFRecords to predict, can use wildcards")
tf.flags.DEFINE_string("label_file", "data/dataset/class_zh.txt", "File containing output labels")
tf.flags.DEFINE_integer("num_labels", None, "Number of output labels")
tf.flags.DEFINE_string("vocab_file", "data/tfrecords/train.vocab", "Vocabulary file, one word per line")
tf.flags.DEFINE_integer("vocab_size", None, "Number of words in vocabulary")
tf.flags.DEFINE_integer("num_oov_vocab_buckets", 20,
                        "Number of hash buckets to use for OOV words")
tf.flags.DEFINE_string("model_dir", "E:\\zsm\\fasttext_tf\\ckpt",
                       "Output directory for checkpoints and summaries")
tf.flags.DEFINE_string("export_dir", "model", "Directory to store savedmodel")

tf.flags.DEFINE_integer("embedding_dimension", 10, "Dimension of word embedding")
tf.flags.DEFINE_boolean("use_ngrams", False, "Use character ngrams in embedding")
tf.flags.DEFINE_integer("num_ngram_buckets", 1000000,
                        "Number of hash buckets for ngrams")
tf.flags.DEFINE_integer("ngram_embedding_dimension", 10, "Dimension of word embedding")

tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate for training")
tf.flags.DEFINE_float("clip_gradient", 5.0, "Clip gradient norm to this ratio")
tf.flags.DEFINE_integer("batch_size", 256, "Training minibatch size")
tf.flags.DEFINE_integer("train_steps", 10000,
                        "Number of train steps, None for continuous")
tf.flags.DEFINE_integer("eval_steps", 100, "Number of eval steps")
tf.flags.DEFINE_integer("num_epochs", None, "Number of training data epochs")
tf.flags.DEFINE_integer("checkpoint_steps", 100,
                        "Steps between saving checkpoints")
tf.flags.DEFINE_integer("num_threads", 4, "Number of reader threads")
tf.flags.DEFINE_boolean("log_device_placement", False, "log where ops are located")
tf.flags.DEFINE_boolean("horovod", False,
                        "Run across multiple GPUs using Horovod MPI. https://github.com/uber/horovod")
tf.flags.DEFINE_boolean("debug", False, "Debug")
FLAGS = tf.flags.FLAGS

if FLAGS.horovod:
    try:
        import horovod.tensorflow as hvd
    except ImportError as e:
        print(e)
        print("Make sure Horovod is installed: https://github.com/uber/horovod")
        sys.exit(1)
    hvd.init()


def InputFn(mode, input_file):
    """
    :param mode: tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT
    :param input_file:  tfrecords
    :return:
    """
    return inputs.InputFn(
        mode, FLAGS.use_ngrams, input_file, FLAGS.vocab_file, FLAGS.vocab_size,
        FLAGS.embedding_dimension, FLAGS.num_oov_vocab_buckets,
        FLAGS.label_file, FLAGS.num_labels,
        FLAGS.ngram_embedding_dimension, FLAGS.num_ngram_buckets,
        FLAGS.batch_size, FLAGS.num_epochs, FLAGS.num_threads)


def Exports(probs, embedding):
    exports = {
        "proba": tf.estimator.export.ClassificationOutput(scores=probs),
        "embedding": tf.estimator.export.RegressionOutput(value=embedding),
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            tf.estimator.export.ClassificationOutput(scores=probs),
    }
    return exports


def FastTextEstimator(model_dir, config=None):
    params = {
        "learning_rate": FLAGS.learning_rate,  # 0.001
    }

    def model_fn(features, labels, mode, params):
        features["text"] = tf.sparse_tensor_to_dense(features["text"],
                                                     default_value=" ")
        if FLAGS.use_ngrams:  # False
            features["ngrams"] = tf.sparse_tensor_to_dense(features["ngrams"],
                                                           default_value=" ")
        text_lookup_table = contrib.lookup.index_table_from_file(
            FLAGS.vocab_file, FLAGS.num_oov_vocab_buckets, FLAGS.vocab_size)
        text_ids = text_lookup_table.lookup(features["text"])
        text_embedding_w = tf.Variable(tf.random_uniform(
            [FLAGS.vocab_size + FLAGS.num_oov_vocab_buckets, FLAGS.embedding_dimension],
            -0.1, 0.1))
        text_embedding = tf.reduce_mean(tf.nn.embedding_lookup(
            text_embedding_w, text_ids), axis=-2)  # 把各个单词的embedding取平均，就是文本的embedding
        input_layer = text_embedding
        if FLAGS.use_ngrams:
            ngram_hash = tf.string_to_hash_bucket(features["ngrams"],
                                                  FLAGS.num_ngram_buckets)
            ngram_embedding_w = tf.Variable(tf.random_uniform(
                [FLAGS.num_ngram_buckets, FLAGS.ngram_embedding_dimension], -0.1, 0.1))
            ngram_embedding = tf.reduce_mean(tf.nn.embedding_lookup(
                ngram_embedding_w, ngram_hash), axis=-2)
            ngram_embedding = tf.expand_dims(ngram_embedding, -2)
            input_layer = tf.concat([text_embedding, ngram_embedding], -1)
        num_classes = FLAGS.num_labels
        logits = contrib.layers.fully_connected(
            inputs=input_layer, num_outputs=num_classes,
            activation_fn=None)
        predictions = tf.argmax(logits, axis=-1)
        probs = tf.nn.softmax(logits)
        loss, train_op = None, None
        metrics = {}
        if mode != tf.estimator.ModeKeys.PREDICT:
            label_lookup_table = contrib.lookup.index_table_from_file(
                FLAGS.label_file, vocab_size=FLAGS.num_labels)
            labels = label_lookup_table.lookup(labels)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits))
            opt = tf.train.AdamOptimizer(params["learning_rate"])
            if FLAGS.horovod:
                opt = hvd.DistributedOptimizer(opt)
            train_op = opt.minimize(loss, global_step=tf.train.get_global_step())
            metrics = {
                "accuracy": tf.metrics.accuracy(labels, predictions)
            }
        exports = {}
        if FLAGS.export_dir:
            exports = Exports(probs, text_embedding)
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, loss=loss, train_op=train_op,
            eval_metric_ops=metrics, export_outputs=exports)

    session_config = tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement)
    if FLAGS.horovod:
        session_config.gpu_options.visible_device_list = str(hvd.local_rank())
    config = contrib.learn.RunConfig(
        save_checkpoints_secs=None,
        save_checkpoints_steps=FLAGS.checkpoint_steps,
        session_config=session_config)
    return tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir,
                                  params=params, config=config)


def FastTrain():
    print("FastTrain", FLAGS.train_steps)
    estimator = FastTextEstimator(FLAGS.model_dir)  # 模型输出地址
    print("TEST" + FLAGS.train_records)
    train_input = InputFn(tf.estimator.ModeKeys.TRAIN, FLAGS.train_records)
    print("STARTING TRAIN")
    hooks = None
    if FLAGS.horovod:
        hooks = [hvd.BroadcastGlobalVariablesHook(0)]
    estimator.train(input_fn=train_input, steps=FLAGS.train_steps, hooks=hooks)
    print("TRAIN COMPLETE")
    if not FLAGS.horovod or hvd.rank() == 0:
        print("EVALUATE")
        eval_input = InputFn(tf.estimator.ModeKeys.EVAL, FLAGS.eval_records)
        # eval_metrics = { "accuracy": tf.metrics.accuracy(labels, predictions) }
        result = estimator.evaluate(input_fn=eval_input, steps=FLAGS.eval_steps, hooks=None)
        print(result)
        print("DONE")
        if FLAGS.export_dir:
            print("EXPORTING")
            estimator.export_savedmodel(FLAGS.export_dir,
                                        inputs.ServingInputFn(FLAGS.use_ngrams))


def main(_):
    if not FLAGS.vocab_size:  # 词汇表里面的单词个数
        FLAGS.vocab_size = len(open(FLAGS.vocab_file, encoding="utf-8").readlines())
    if not FLAGS.num_labels:  # 标签个数
        FLAGS.num_labels = len(open(FLAGS.label_file, encoding="utf-8").readlines())
    if FLAGS.horovod:
        nproc = hvd.size()
        total = FLAGS.train_steps
        FLAGS.train_steps = total / nproc
        print("Running %d steps on each of %d processes for %d total" % (
            FLAGS.train_steps, nproc, total))
    FastTrain()


if __name__ == '__main__':
    if FLAGS.debug:
        tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
