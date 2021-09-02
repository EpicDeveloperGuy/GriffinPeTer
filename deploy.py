import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import dataset

import model
from model import TransformerModel

from torchtext.data.datasets_utils import _RawTextIterableDataset
from torchtext.data.datasets_utils import _read_text_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

dream = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trans = torch.load('peter.pth').to(device)

train_iter = _RawTextIterableDataset('PETER', 394, _read_text_iterator('peter.data'))
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

def predict(model: nn.Module, prefix: str, length: int):
    trans.eval()
    src_mask = generate_square_subsequent_mask(dream).to(device)
    for i in range(length):
        with torch.no_grad():
            input = data_process(prefix).to(device)[-2:]
            output = trans(input, src_mask)
            output_dist = output.data.view(-1).div(1).exp()
            top_i = torch.multinomial(output_dist, 1)[0]
            weird = vocab.lookup_token(top_i/4)
            if (weird != '<unk>'):
                prefix += ' ' + weird
    return prefix

print(predict(trans, 'Hey Lois', 20))