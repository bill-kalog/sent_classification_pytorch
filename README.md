## Description
model variants for text classification written in [PyTorch](http://pytorch.org/).
Some of the code structure is based on examples from [spro](https://github.com/spro/practical-pytorch).
At the moment the following models are implemented:
* vanilla RNN with GRU cells `class RNN_s`
* vanilla RNN with GRU cells and attention on top `class RNN_encoder`

### install torchtext
in order to run this code apart from [PyTorch](http://pytorch.org/) you need [torchtext](https://github.com/pytorch/text/tree/master/torchtext) too.
Get it using:
```bash
pip install torchtext
```

### use tensorboard on pytorch

check the [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) repo.
Here are is a  [blogpost](https://medium.com/@dexterhuang/tensorboard-for-pytorch-201a228533c5)
describing some of the functionality.


