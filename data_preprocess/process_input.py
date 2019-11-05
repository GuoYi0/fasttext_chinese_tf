# -*- coding:utf-8 -*-
"""Process input data into tensorflow examples, to ease training.

Input data is in one of two formats:
- facebook's format used in their fastText library.
- two text files, one with input text per line, the other a label per line.
"""

import os.path
import re
import sys
import tensorflow as tf
import inputs
import text_utils
from collections import Counter


tf.flags.DEFINE_string("facebook_input", None,
                       "Input file in facebook train|test format")
tf.flags.DEFINE_string("text_input", None,
                       """Input text file containing one text phrase per line.
                       Must have --labels defined
                       Used instead of --facebook_input""")
tf.flags.DEFINE_string("class_file", "../data/dataset/class_zh.txt", "class file")
tf.flags.DEFINE_string("text_and_label_input", "../data/dataset/test.txt", "text and label exist in one line")
tf.flags.DEFINE_string("stopwords_file", "../data/stopwords/stopwords.txt", "the file containing chinese stop words")
tf.flags.DEFINE_string("labels", None,
                       """Input text file containing one label for
                       classification  per line.
                       Must have --text_input defined.
                       Used instead of --facebook_input""")
tf.flags.DEFINE_string("ngrams", None,
                       "list of ngram sizes to create, e.g. --ngrams=2,3,4,5")
tf.flags.DEFINE_string("output_dir", "../data/tfrecords",
                       "Directory to store resulting vector models and checkpoints in")
tf.flags.DEFINE_integer("num_shards", 1,
                        "Number of outputfiles to create")
FLAGS = tf.flags.FLAGS


def ParseFacebookInput(inputfile, ngrams):
    """Parse input in the format used by facebook FastText.
    labels are formatted as __label__1
    where the label values start at 0.
    """
    examples = []
    for line in open(inputfile):
        words = line.split()
        # label is first field with __label__ removed
        match = re.match(r'__label__(.+)', words[0])
        label = match.group(1) if match else None
        # Strip out label and first ,
        first = 2 if words[1] == "," else 1
        words = words[first:]
        examples.append({
            "text": words,
            "label": label
        })
        if ngrams:
            examples[-1]["ngrams"] = text_utils.GenerateNgrams(words, ngrams)
    return examples


def ParseTextInput(textfile, labelsfile, ngrams):
    """Parse input from two text files: text and labels.
    labels are specified 0-offset one per line.
    """
    examples = []
    with open(textfile) as f1, open(labelsfile) as f2:
        for text, label in zip(f1, f2):
            words = text_utils.TokenizeText(text)
            examples.append({
                "text": words,
                "label": label,
            })
            if ngrams:
                examples[-1]["ngrams"] = text_utils.GenerateNgrams(words, ngrams)
    return examples


def ParseText_and_label_Input(textfile, labelsmap, stopwords_file, ngrams):
    """从两个文件中解析数据，一个是训练集。一行是一条数据，用"""
    idx_to_textlabel = {}
    with open(labelsmap, encoding='utf-8') as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        idx_to_textlabel[idx] = line.strip()
    examples = []
    with open(stopwords_file, encoding="utf-8") as sf:
        lines = sf.readlines()
    stop_words = set([line.strip() for line in lines])
    stop_words.add(' ')
    stop_words.add('\n')
    stop_words.add('')
    with open(textfile, encoding='utf-8') as f1:
        for record in f1:
            text, label = record.strip().split('	')
            text = text.strip()
            label = idx_to_textlabel[int(label.strip())]
            words = text_utils.TokenizeText(text, stop_words)
            examples.append({
                "text": words,
                "label": label,
            })
            if ngrams:
                examples[-1]["ngrams"] = text_utils.GenerateNgrams(words, ngrams)
    return examples


def WriteExamples(examples, outputfile, num_shards):
    """Write examles in TFRecord format.
    Args:
      examples: list of feature dicts.
                {'text': [words], 'label': [labels]}
      outputfile: full pathname of output file
    """
    shard = 0
    num_per_shard = len(examples) / num_shards + 1
    for n, example in enumerate(examples):
        if n % num_per_shard == 0:
            shard += 1
            writer = tf.python_io.TFRecordWriter(outputfile + '-%d-of-%d' % (shard, num_shards))
        record = inputs.BuildTextExample(
            example["text"], example.get("ngrams", None), example["label"])
        writer.write(record.SerializeToString())


def WriteVocab(examples, vocabfile, labelfile):
    words = Counter()  # 定义一个计数器。键是词，值是这个词的频次
    labels = set()
    for example in examples:
        words.update(example["text"])
        labels.add(example["label"])
    with open(vocabfile, "w", encoding="utf-8") as f:
        # Write out vocab in most common first order
        # We need this as NCE loss in TF uses Zipf distribution
        for word in words.most_common():
            f.write(word[0] + '\n')
    with open(labelfile, "w", encoding="utf-8") as f:
        labels = sorted(list(labels))
        for label in labels:
            f.write(str(label) + '\n')


def main(_):
    # Check flags
    if not (FLAGS.facebook_input or (FLAGS.text_input and FLAGS.labels) or (FLAGS.text_and_label_input and FLAGS.class_file)):
        sys.stderr.write("Error: You must define either facebook_input or both text_input and labels")
        sys.exit(1)
    ngrams = None
    if FLAGS.ngrams:
        ngrams = text_utils.ParseNgramsOpts(FLAGS.ngrams)
    if FLAGS.facebook_input:
        inputfile = FLAGS.facebook_input
        examples = ParseFacebookInput(FLAGS.facebook_input, ngrams)
    elif FLAGS.text_input:
        inputfile = FLAGS.text_input
        examples = ParseTextInput(FLAGS.text_input, FLAGS.labels, ngrams)
    else:
        inputfile = FLAGS.text_and_label_input
        examples = ParseText_and_label_Input(inputfile, FLAGS.class_file, FLAGS.stopwords_file, ngrams)

    file_name = str(os.path.basename(inputfile).split('.')[0])
    outputfile = os.path.join(FLAGS.output_dir, file_name + ".tfrecords")
    WriteExamples(examples, outputfile, FLAGS.num_shards)
    # 词汇表。把输入数据分词以后，按照单词的频次从高到低输出
    vocabfile = os.path.join(FLAGS.output_dir, file_name + ".vocab")
    labelfile = os.path.join(FLAGS.output_dir, file_name + ".labels")
    WriteVocab(examples, vocabfile, labelfile)


if __name__ == '__main__':
    tf.app.run()
