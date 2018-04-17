## Description
model variants for text classification written in [PyTorch](http://pytorch.org/).
Some of the code structure is based on examples from [spro](https://github.com/spro/practical-pytorch).
At the moment the following models are implemented:
* vanilla RNN with GRU cells `class RNN_s`
* vanilla RNN with GRU cells and attention on top `class RNN_encoder`
* CNN with attention on top `class CNN_encoder`
* a transformer encoder in `transformer_models.py`

### install torchtext
in order to run this code apart from [PyTorch](http://pytorch.org/) you need [torchtext](https://github.com/pytorch/text/tree/master/torchtext) too.
Get it using:
```bash
pip install torchtext
```

### tokenization
if you want to use some more advanced tokenization technique with [torchtext](https://github.com/pytorch/text/tree/master/torchtext) do make sure
you have [spacy](https://spacy.io/) installed using:
```bash
pip install -U spacy
```
then download the English models using:
```bash
python -m spacy download en
```

### use tensorboard on pytorch

check the [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) repo.
Here are is a  [blogpost](https://medium.com/@dexterhuang/tensorboard-for-pytorch-201a228533c5)
describing some of the functionality.


