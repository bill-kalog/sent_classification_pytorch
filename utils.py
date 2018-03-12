from torchtext import data
from torchtext import datasets


def load_data(chosen_dataset, chosen_dataset_name):
    '''
    load data, split dataset and create vocabulary
    '''
    inputs = data.Field(
        lower=True, include_lengths=True, batch_first=True, tokenize='spacy')
    answers = data.Field(
        sequential=False)

    if chosen_dataset_name == 'SST_SENT':
        train, dev, test = chosen_dataset.splits(inputs, answers)
    else:
        train, dev, test = data.TabularDataset.splits(
            path='.data/ReParse_first_batch', train='train_firstBatch.csv',
            validation='valid_firstBatch.csv', test='test_firstBatch.csv',
            format='csv', fields=[('label', answers), ('text', inputs)])

    print('Building vocabulary')
    inputs.build_vocab(train, dev, test)
    inputs.vocab.load_vectors('glove.6B.300d')

    answers.build_vocab(train)
    return train, dev, test, inputs, answers
