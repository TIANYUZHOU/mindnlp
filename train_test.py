import math
import numpy as np

import mindspore as ms
from mindspore.common.initializer import Uniform, HeUniform
from mindspore import nn, ops, Tensor
from mindspore.dataset import text
from tqdm import tqdm

from mindnlp.abc import Seq2vecModel
from mindnlp.dataset import CoNLL2000Chunking, CoNLL2000Chunking_Process
from mindnlp.engine.trainer import Trainer
from mindnlp.modules import CRF, RNNEncoder

dataset_train,dataset_test = CoNLL2000Chunking()

vocab = text.Vocab.from_dataset(dataset_train,columns=["words"],freq_range=None,top_k=None,
                                   special_tokens=["<pad>","<unk>"],special_first=True)


dataset_train = CoNLL2000Chunking_Process(dataset=dataset_train, vocab=vocab, batch_size=32, max_len=80)

class Head(nn.Cell):
    """ Head for BiLSTM-CRF model """
    def __init__(self, hidden_dim, num_tags):
        super().__init__()
        weight_init = HeUniform(math.sqrt(5))
        bias_init = Uniform(1 / math.sqrt(hidden_dim * 2))
        self.hidden2tag = nn.Dense(hidden_dim, num_tags, weight_init=weight_init, bias_init=bias_init)
        # self.hidden2tag = nn.Dense(hidden_dim, num_tags)

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
        feats = self.head(output)
        res = self.crf(feats, label, seq_length)
        return res

embedding_dim = 16
hidden_dim = 32
embedding = nn.Embedding(vocab_size=len(vocab.vocab()), embedding_size=embedding_dim, padding_idx=vocab.tokens_to_ids("<pad>"))
lstm_layer = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
encoder = RNNEncoder(embedding, lstm_layer)
head = Head(hidden_dim, 23)
net = BiLSTM_CRF(encoder, head, 23)

optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01, weight_decay=1e-4)

grad_fn = ops.value_and_grad(net, None, optimizer.parameters)

print(">>>>>>开始训练<<<<<<")

# 使用 triner 训练（魔改过的trainer与主仓不一致）

trainer = Trainer(network=net, train_dataset=dataset_train, eval_dataset=None, metrics=None,
              epochs=5, loss_fn=None, optimizer=optimizer)

trainer.run(tgt_columns="label", jit=False)

# 使用原生训练

# def train_step(data, seq_length, label):
#     """ train_step """
#     loss, grads = grad_fn(data, seq_length, label)
#     # clip = []
#     # for grad in grads:
#     #     new_grad = ops.clip_by_value(grad, Tensor(1.0, ms.float32))
#     #     clip.append(new_grad)
#     grads = ops.clip_by_global_norm(grads, clip_norm=1.0)
#     loss = ops.depend(loss, optimizer(grads))
#     return loss

# size = dataset_train.get_dataset_size()

# for batch, (data, seq_length, label) in enumerate(dataset_train.create_tuple_iterator()):
#     loss = train_step(data, seq_length ,label)
#     loss, current = loss.asnumpy(), batch
#     print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")
#     # print("data:", data)
#     # print("label:", label)
#     # print("seq_length:", seq_length)
#     if str(loss) == "nan":
#         break