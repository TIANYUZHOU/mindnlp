# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

""" Sentiment Analysis Model """

# pylint: disable=arguments-differ

import os

import mindspore
# from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.dataset import text,GeneratorDataset,transforms
from mindspore import nn
from mindnlp.engine.callbacks import CheckpointCallback
from mindnlp.transforms import PadTransform
from mindnlp.models import BertModel, BertConfig
from mindnlp.transforms.tokenizers import BertTokenizer
from mindnlp.engine import Trainer, Accuracy

class SentimentDataset:
    """Sentiment Dataset"""
    def __init__(self, path):
        self.path = path
        self._labels, self._text_a= [], []
        self._load()

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as dataset_file:
            dataset = dataset_file.read()
        lines = dataset.split("\n")
        for line in lines[1:-1]:
            label, text_a = line.split("\t")
            self._labels.append(int(label))
            self._text_a.append(text_a)

    def __getitem__(self, index):
        return self._labels[index], self._text_a[index]

    def __len__(self):
        return len(self._labels)

column_names = ["label", "text_a"]

dataset_train = GeneratorDataset(source=SentimentDataset("data/train.tsv"),
                                 column_names=column_names, shuffle=False)

dataset_val = GeneratorDataset(source=SentimentDataset("data/dev.tsv"),
                               column_names=column_names, shuffle=False)

dataset_test = GeneratorDataset(source=SentimentDataset("data/test.tsv"),
                                column_names=column_names, shuffle=False)

vocab_path = os.path.join("/home/tianyuzhou/.mindnlp/workflow/sentiment_analysis/bert", "vocab.txt")
vocab = text.Vocab.from_file(vocab_path)

vocab_size = len(vocab.vocab())

pad_value_text = vocab.tokens_to_ids('[PAD]')

tokenizer = BertTokenizer(vocab=vocab)
pad_op_text = PadTransform(max_length=128, pad_value=pad_value_text)
type_cast_op = transforms.TypeCast(mindspore.int32)

dataset_train = dataset_train.map(operations=[tokenizer,pad_op_text], input_columns="text_a")
dataset_train = dataset_train.map(operations=[type_cast_op], input_columns="label")

dataset_val = dataset_val.map(operations=[tokenizer,pad_op_text], input_columns="text_a")
dataset_val = dataset_val.map(operations=[type_cast_op], input_columns="label")

rename_columns = ["label", "input_ids"]
dataset_train = dataset_train.rename(input_columns=column_names, output_columns=rename_columns)
dataset_val = dataset_val.rename(input_columns=column_names, output_columns=rename_columns)

dataset_train = dataset_train.batch(1)
dataset_val = dataset_test.batch(1)

class BertForSequenceClassification(nn.Cell):
    """Bert Model for classification tasks"""
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        mindspore.load_param_into_net(self.bert, state_dict)
        self.classifier = nn.Dense(config.hidden_size, config.num_labels)

    def construct(self, input_ids, attention_mask=None, token_type_ids=None, \
        position_ids=None, head_mask=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask
        )
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        return logits

model_path = os.path.join("checkpoints/bert-base-chinese.ckpt")
state_dict = mindspore.load_checkpoint(model_path)

config = BertConfig(vocab_size=vocab_size, num_labels=3)
model_instance = BertForSequenceClassification(config)

loss = nn.CrossEntropyLoss()
optimizer = nn.Adam(model_instance.trainable_params(), learning_rate=1e-5)

metric = Accuracy()

ckpoint_cb = CheckpointCallback(save_path='sentimentbert_ckpt',epochs=1 ,keep_checkpoint_max=10)

trainer = Trainer(network=model_instance, train_dataset=dataset_train,
                  eval_dataset=dataset_val, metrics=metric,
                  epochs=10, loss_fn=loss, optimizer=optimizer,callbacks=[ckpoint_cb])
trainer.run(tgt_columns="label")
