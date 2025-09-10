# PyTorch implementation of minimal RNNs

PyTorch implementation of Minimal RNNs (minGRU and minLSTM) as proposed in [Feng et al. 2024, arXiv preprint](https://arxiv.org/abs/2410.01201).

## Install
```sh
# Clone this repository
git clone https://github.com/rnkj/minrnn.git
cd minrnn

# Prepare Python environment
python3 -m venv .venv
source .venv venv

# Install PyTorch
pip install torch torchvision

# Install this package
pip install -e .
```

## Usage
### Example
```python
import torch
from minrnn import MinGRUCell
# from minrnn import MinLSTMCell

batch_size, seq_length, input_size, hidden_size = 32, 100, 64, 128
x = torch.randn(batch_size, seq_length, input_size)
cell = MinGRUCell(input_size, hidden_size)
# cell = MinLSTMCell(input_size, hidden_size)

# Forward pass (training mode)
cell.train()
output = cell(x)
```

### Runtime test
```
python -m minrnn.runtime_test -d cuda
```

## License

MIT License

## Reference

[Minimal RNNs: Efficient and Interpretable Recurrent Neural Networks](https://arxiv.org/abs/2410.01201)

```
@article{feng2024minimal,
    title={Minimal RNNs: Efficient and Interpretable Recurrent Neural Networks},
    author={Feng, Y. and Nakajo, R. and Ogata, T.},
    journal={arXiv preprint arXiv:2410.01201},
    year={2024}
}
```