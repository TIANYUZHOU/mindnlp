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
"""BiLSTM-CRF model"""
# pylint: disable=abstract-method,invalid-name,arguments-differ

import numpy as np
import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.text as text
from tqdm import tqdm
from mindspore import nn,ops
from mindspore.dataset.text.utils import Vocab
from mindnlp.engine.metrics.accuracy import Accuracy
from mindnlp.modules import CRF,Word2vec,RNNEncoder,embeddings
from mindnlp.abc import Seq2vecModel
from mindnlp.engine.trainer import Trainer


class Head(nn.Cell):
    """ Head for BiLSTM-CRF model """
    def __init__(self, hidden_dim, num_tags):
        super().__init__()
        self.hidden2tag = nn.Dense(hidden_dim, num_tags, 'he_uniform')

    def construct(self, context):
        return self.hidden2tag(context)

class BiLSTM_CRF(Seq2vecModel):
    """ BiLSTM-CRF model """
    def __init__(self, encoder, head, num_tags):
        super().__init__(encoder, head)
        self.encoder = encoder
        self.head = head
        self.crf = CRF(num_tags, batch_first=True)

    def construct(self, text, seq_length, label=None):
        output,_,_ = self.encoder(text)
        # a = self.encoder(text)
        # print(a)
        feats = self.head(output)
        res = self.crf(feats, label, seq_length)
        return res

class TestDataset:
    """ TsetDataset """
    def __init__(self):
        self._text = []
        self._label = []
        self._seq_length = []
        self._load()

    def _load(self):
        _text = ["清 华 大 学 坐 落 于 首 都 北 京".split(),"重 庆 是 一 个 魔 幻 城 市".split()]
        _label = ["B I I I O O O O O B I".split(),"B I O O O O O O O".split()]
        self._text.extend(_text)
        self._label.extend(_label)
        for item in _text:
            self._seq_length.append(len(item))

    def __getitem__(self, index):
        return self._text[index], self._label[index],self._seq_length[index]

    def __len__(self):
        return len(self._text)

if __name__ == '__main__':

    dataset = ds.GeneratorDataset(TestDataset(),column_names=["text","label","seq_length"],shuffle=False)

    # itr = dataset.create_dict_iterator()

    # print(list(itr))
    # vocab = text.Vocab.from_list("清 华 大 学 坐 落 于 首 都 北 京".split()+"重 庆 是 一 个 魔 幻 城 市".split(),special_tokens=["<pad>"])
    vocab = text.Vocab.from_dataset(dataset,columns=["text"],freq_range=None,top_k=None,
                                   special_tokens=["<pad>","<unk>"],special_first=True)

    lookup_op = ds.text.Lookup(vocab,unknown_token='<unk>')
    # lookup_op = ds.text.Lookup(vocab)
    pad_op_text = ds.transforms.PadEnd([80],pad_value=vocab.tokens_to_ids('<pad>'))
    pad_op_label = ds.transforms.PadEnd([80],pad_value=2)

    tag_to_idx = {"B": 0, "I": 1, "O": 2}
    def tag_idx(tags):
        """ tag_idx """
        tag_idx_list = []
        for tag in tags:
            tag_idx_list.append(tag_to_idx[tag])
        return tag_idx_list

    type_cast_op = ds.transforms.TypeCast(ms.int64)

    dataset = dataset.map(operations=[lookup_op,pad_op_text,type_cast_op],input_columns=['text'])
    dataset = dataset.map(operations=[tag_idx],input_columns=['label'])
    dataset = dataset.map(operations=[pad_op_label],input_columns=['label'])
    dataset = dataset.map(operations=[type_cast_op],input_columns=['label'])
    dataset = dataset.map(operations=[type_cast_op],input_columns=['seq_length'])
    dataset = dataset.batch(2)

    # itr = dataset.create_tuple_iterator()
    # # print(list(itr))
    # data_list = list(itr)
    # seqs = ops.stack([data_list[0][0], data_list[1][0]])
    # labels = ops.stack([data_list[0][1], ops.concat((data_list[1][1],ms.Tensor([2,2],ms.int64)))])
    # seq_lens = ops.stack((data_list[0][2], data_list[1][2]))
    # print("data:",seqs)
    # print("label:",labels)
    # print("seq_length:",seq_lens)

    # print(vocab)

    embedding_dim = 16
    hidden_dim = 32
    embedding = nn.Embedding(vocab_size=len(vocab.vocab()), embedding_size=embedding_dim, padding_idx=vocab.tokens_to_ids('<pad>'))
    lstm_layer = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
    encoder = RNNEncoder(embedding, lstm_layer)
    head = Head(hidden_dim, len(tag_to_idx))
    net = BiLSTM_CRF(encoder, head, len(tag_to_idx))

    # loss = nn.NLLLoss()

    # # define metrics
    # metric = Accuracy()

    optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01, weight_decay=1e-4)

    # trainer = Trainer(network=net, train_dataset=dataset, eval_dataset=None, metrics=None,
    #               epochs=5, loss_fn=loss, optimizer=optimizer)
    # trainer.run(tgt_columns="label", jit=False)

    grad_fn = ops.value_and_grad(net, None, optimizer.parameters)

    def train_step(data, seq_length, label):
        """ train_step """
        loss, grads = grad_fn(data, seq_length, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    itr = dataset.create_tuple_iterator()
    seqs, labels, seq_lens = next(itr)
    print("seqs:",seqs)
    print("labels:",labels)
    print("seq_lens:",seq_lens)
    steps = 500
    with tqdm(total=steps) as t:
        for i in range(steps):
            loss = train_step(seqs, seq_lens, labels)
            t.set_postfix(loss=loss)
            t.update(1)
            
        