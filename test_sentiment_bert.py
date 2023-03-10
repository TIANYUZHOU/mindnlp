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
"""Test """

import numpy as np

import mindspore

from mindspore import Tensor
from mindspore import context
from mindspore.ops import functional as F

from mindnlp.models.bert import BertConfig
from mindnlp.workflow import BertForSequenceClassification

label_map = {0: "negative", 1: "positive"}
context.set_context(mode=context.PYNATIVE_MODE)
config = BertConfig(num_labels=2)
model = BertForSequenceClassification(config)

input_ids = Tensor(np.random.randn(2, 512), mindspore.int32)

outputs = model(input_ids)
probs = F.softmax(outputs, axis=1)
print(probs)
print(probs.shape)
score = [max(prob) for prob in probs]
print(score)
idx = [1,1]
labels = [label_map[i] for i in idx]
print(labels)
