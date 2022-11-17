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
# pylint: disable=arguments-differ

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from mindspore.dataset.text.utils import Vocab
from mindnlp.modules import CRF,Word2vec,RNNEncoder,embeddings
from mindnlp.abc import Seq2vecModel


# class BiLSTM_CRF(nn.Cell):
#     """ BiLSTM-CRF model """
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, padding_idx=0):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
#         self.rnn_encoder = RNNEncoder(self.embedding, self.lstm)
#         self.seq2vec = Seq2vecModel(self.rnn_encoder,self.embedding)
#         self.hidden2tag = nn.Dense(hidden_dim, num_tags, 'he_uniform')
#         self.crf = CRF(num_tags, batch_first=True)

#     def construct(self, inputs, seq_length, tags=None):
#         embeds = self.embedding(inputs)
#         outputs, _ = self.lstm(embeds, seq_length=seq_length)
#         feats = self.hidden2tag(outputs)

#         crf_outs = self.crf(feats, tags, seq_length)
#         return crf_outs

# class BiLSTM_CRF(nn.Cell):
#     """ BiLSTM-CRF model """
#     def __init__(self, vocab, init_embed, num_tags, embedding_dim, hidden_dim):
#         super().__init__()
#         self.embedding = Word2vec(vocab=vocab, init_embed=init_embed)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
#         self.hidden2tag = nn.Dense(hidden_dim, num_tags, 'he_uniform')
#         self.crf = CRF(num_tags, batch_first=True)

#     def construct(self, inputs, seq_length, tags=None):
#         embeds = self.embedding(inputs)
#         outputs, _ = self.lstm(embeds, seq_length=seq_length)
#         feats = self.hidden2tag(outputs)

#         crf_outs = self.crf(feats, tags, seq_length)
#         return crf_outs


class BiLSTM_CRF(nn.Cell):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Dense(hidden_dim, num_tags, 'he_uniform')
        self.crf = CRF(num_tags, batch_first=True)

    def construct(self, inputs, seq_length, tags=None):
        embeds = self.embedding(inputs)
        # print("embeds.shape:",embeds.shape)
        outputs, _ = self.lstm(embeds, seq_length=seq_length)
        feats = self.hidden2tag(outputs)

        crf_outs = self.crf(feats, tags, seq_length)
        return crf_outs


embedding_dim = 16
hidden_dim = 32

training_data = [(
    "清 华 大 学 坐 落 于 首 都 北 京".split(),
    "B I I I O O O O O B I".split()
), (
    "重 庆 是 一 个 魔 幻 城 市".split(),
    "B I O O O O O O O".split()
)]



word_to_idx = {}
word_to_idx['<pad>'] = 0
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

tag_to_idx = {"B": 0, "I": 1, "O": 2}

vocab = Vocab.from_list("清 华 大 学 坐 落 于 首 都 北 京".split()+"重 庆 是 一 个 魔 幻 城 市".split())
init_embed = Tensor(np.zeros((len(word_to_idx), embedding_dim)).astype(np.float32))

model = BiLSTM_CRF(len(word_to_idx), embedding_dim, hidden_dim, len(tag_to_idx))
# model = BiLSTM_CRF(vocab=vocab, init_embed=init_embed,embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_tags = len(tag_to_idx))
optimizer = nn.SGD(model.trainable_params(), learning_rate=0.01, weight_decay=1e-4)


grad_fn = ops.value_and_grad(model, None, optimizer.parameters)

def train_step(data, seq_length, label):
    loss, grads = grad_fn(data, seq_length, label)
    loss = ops.depend(loss, optimizer(grads))
    return loss


def prepare_sequence(seqs, word_to_idx, tag_to_idx):
    seq_outputs, label_outputs, seq_length = [], [], []
    max_len = max([len(i[0]) for i in seqs])

    for seq, tag in seqs:
        seq_length.append(len(seq))
        idxs = [word_to_idx[w] for w in seq]
        labels = [tag_to_idx[t] for t in tag]
        idxs.extend([word_to_idx['<pad>'] for i in range(max_len - len(seq))])
        labels.extend([tag_to_idx['O'] for i in range(max_len - len(seq))])
        seq_outputs.append(idxs)
        label_outputs.append(labels)
    # print("seq_outputs:",seq_outputs)
    # print("label_outputs:",label_outputs)
    # print("seq_length:",seq_length)

    return ms.Tensor(seq_outputs, ms.int64), \
            ms.Tensor(label_outputs, ms.int64), \
            ms.Tensor(seq_length, ms.int64)


data, label, seq_length = prepare_sequence(training_data, word_to_idx, tag_to_idx)

# print("data:",type(data))
# print("label:",label)
# print("seq_length:",seq_length)

from tqdm import tqdm

steps = 500
with tqdm(total=steps) as t:
    for i in range(steps):
        loss = train_step(data, seq_length, label)
        t.set_postfix(loss=loss)
        t.update(1)
