import os
import glob

# from .. import data
from torchtext import data
from my_example import ExampleMy

flag_phrases = True
# use code from:
# https://github.com/pytorch/text/blob/master/torchtext/datasets/sst.py


class SST_SENT(data.Dataset):
    '''
    load SST finegrained version from
    https://github.com/harvardnlp/sent-conv-torch
    '''
    # links without phrases
    # urls = ['https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/data/stsa.fine.dev', 'https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/data/stsa.fine.test', 'https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/data/stsa.fine.train']
    # links with phrases on train set
    urls = ['https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/data/stsa.fine.dev', 'https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/data/stsa.fine.test', 'https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/data/stsa.fine.phrases.train']

    name = 'SST_SENTIMENT'
    dirname = ''

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, subtrees=False,
                 fine_grained=True, **kwargs):
        """Create an SST dataset instance given a path and fields.
        Arguments:
            path: Path to the data file
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            subtrees: Whether to include sentiment-tagged subphrases
                in addition to complete examples. Default: False.
            fine_grained: Whether to use 5-class instead of 3-class
                labeling. Default: True.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        fields = [('text', text_field), ('label', label_field)]
        print ('loading file: {}'.format(path))

        def get_label_str(label):
            pre = 'very ' if fine_grained else ''
            return {'0': pre + 'negative', '1': 'negative', '2': 'neutral',
                    '3': 'positive', '4': pre + 'positive', None: None}[label]
        label_field.preprocessing = data.Pipeline(get_label_str)

        with open(os.path.expanduser(path)) as f:
            if subtrees:
                pass
                # not supported
                examples = [ex for line in f for ex in
                            data.Example.fromCSV(line, fields, True)]
            else:
                # examples = [data.Example.fromCSV(line, fields) for line in f]
                examples = [ExampleMy.fromSplitOnFirst(line, fields) for line in f]
        super(SST_SENT, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, root='.data',
               train='stsa.fine.train', test='stsa.fine.test',
               validation='stsa.fine.dev', **kwargs):
        """Create dataset objects for splits of the SST_SENT dataset.
        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            root: Root dataset storage directory. Default is 'SST_f'.
            train: The directory that contains the training examples
            test: The directory that contains the test examples
            validation: name of dev set
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        # path = cls.download(root)
        if flag_phrases:
            train = 'stsa.fine.phrases.train'
        return super(SST_SENT, cls).splits(
            root=root, text_field=text_field, label_field=label_field,
            train=train, validation=validation, test=test, **kwargs)

    @classmethod
    def iters(cls, batch_size=32, device=None, root='.data', vectors=None, **kwargs):
        """Creater iterator objects for splits of the SST_SENT dataset.
        Arguments:
            batch_size: Batch_size
            device: Device to create batches on. Use - 1 for CPU and None for
                the currently active GPU device.
            root: The root directory that contains the SST_SENT dataset subdirectory
            vectors: one of the available pretrained vectors or a list with each
                element one of the available pretrained vectors (see Vocab.load_vectors)
            Remaining keyword arguments: Passed to the splits method.
        """
        TEXT = data.Field()
        LABEL = data.Field(sequential=False)

        train, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)
        LABEL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, test), batch_size=batch_size, device=device)
