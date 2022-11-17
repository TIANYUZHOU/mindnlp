from mindnlp.dataset.register import process
from mindspore.dataset import GeneratorDataset, text, transforms
import mindspore
import re

from mindnlp.dataset.utils import make_bucket
from mindnlp.dataset.transforms.seq_process import TruncateSequence

@process.register
def CoNLL2000Chunking_Process(dataset, vocab, batch_size=64, max_len=80, \
                 bucket_boundaries=None, drop_remainder=False):
    """
    the process of the IMDB dataset

    Args:
        dataset (GeneratorDataset): IMDB dataset.
        vocab (Vocab): vocabulary object, used to store the mapping of token and index.

    Returns:
        - **dataset** (MapDataset) - dataset after transforms.

    Raises:
        TypeError: If `input_column` is not a string.

    Examples:
    """
    columns_to_project = ["words", "chunk_tag"]
    dataset = dataset.project(columns= columns_to_project)
    input_columns = ["words", "chunk_tag"]
    output_columns = ["text", "label"]
    dataset = dataset.rename(input_columns=input_columns, output_columns=output_columns)

    class TmpDataset:
        """ a Dataset for seq_length column """
        def __init__(self, dataset):
            self._dataset = dataset
            self._seq_length = []
            self._load()

        def _load(self):
            for data in self._dataset.create_dict_iterator():
                self._seq_length.append(len(data["text"]))

        def __getitem__(self, index):
            return self._seq_length[index]

        def __len__(self):
            print(max(self._seq_length), min(self._seq_length))
            return len(self._seq_length)

    dataset_tmp = GeneratorDataset(TmpDataset(dataset), ["seq_length"],shuffle=False)
    dataset = dataset.zip(dataset_tmp)

    pad_value_text = vocab.tokens_to_ids('<pad>')
    pad_value_label = 2
    lookup_op = text.Lookup(vocab, unknown_token='<unk>')
    type_cast_op = transforms.TypeCast(mindspore.int64)

    def tag_idx(tags):
        """ tag_idx """
        tag_idx_list = []
        regex_dic = {"^B.*":0, "^I.*":1,"^O.*":2}
        for tag in tags:
            for key, value in regex_dic.items():
                if re.match(key, tag):
                    tag_idx_list.append(value)
        return tag_idx_list

    dataset = dataset.map(operations=[tag_idx], input_columns=["label"])
    dataset = dataset.map(operations=[lookup_op], input_columns=["text"])

    if bucket_boundaries is not None:
        if not isinstance(bucket_boundaries, list):
            raise ValueError(f"'bucket_boundaries' must be a list of int, but get {type(bucket_boundaries)}")
        trancate_op = TruncateSequence(max_len)
        dataset = dataset.map([trancate_op], 'text')
        if bucket_boundaries[-1] < max_len + 1:
            bucket_boundaries.append(max_len + 1)
        bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)
        dataset = make_bucket(dataset, 'text', pad_value_text, \
                              bucket_boundaries, bucket_batch_sizes, drop_remainder)
        dataset = make_bucket(dataset, 'label', pad_value_label, \
                        bucket_boundaries, bucket_batch_sizes, drop_remainder)
    else:
        pad_op_text = transforms.PadEnd([max_len], pad_value_text)
        pad_op_label = transforms.PadEnd([max_len], pad_value_label)
        dataset = dataset.map([pad_op_text], 'text')
        dataset = dataset.map([pad_op_label], 'label')
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    dataset = dataset.map(operations=[type_cast_op])

    return dataset
    