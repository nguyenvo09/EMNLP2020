import numpy as np
import torch, os
import torch.nn.utils.rnn as rnn_utils
from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision
from torchvision import transforms


def flatten(x):
    '''
    flatten high dimensional tensor x into an array
    :param x: shape (B, D1, D2, ...)
    :return: 1 dimensional tensor
    '''
    dims = x.size()[1:] #remove the first dimension as it is batch dimension
    num_features = 1
    for s in dims: num_features *= s
    return x.contiguous().view(-1, num_features)


def gpu(tensor, gpu=False):

    if gpu: return tensor.cuda()
    else: return tensor


def cpu(tensor):
    if tensor.is_cuda: return tensor.cpu()
    else: return tensor


def minibatch(*tensors, **kwargs):

    batch_size = kwargs['batch_size']

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    """This is not an inplace operation. Therefore, you can shuffle without worrying changing data."""
    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices) # fix this for reproducible

    if len(arrays) == 1:
        return arrays[0][shuffle_indices]
    else:
        return tuple(x[shuffle_indices] for x in arrays)


def assert_no_grad(variable):

    if variable.requires_grad:
        raise ValueError(
            "nn criterions don't compute the gradient w.r.t. targets - please "
            "mark these variables as volatile or not requiring gradients"
        )


def numpy2tensor(x, dtype):
    # torch.tensor(torch.from_numpy(var), dtype = torch.int, torch.long)
    return torch.tensor(torch.from_numpy(x), dtype = dtype)


def tensor2numpy(x):
    # return x.numpy()
    return cpu(x).numpy()


def set_seed(seed, cuda=False):

    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)


def create_mask_tensor(query: torch.Tensor, doc: torch.Tensor, threshold: int = 0):
    """
    Creating masking of two tensor. These two tensors are integer tensor
    Parameters

    ----------
    query: (B, L)
    doc: (B, R)
    threshold: when it is 0, means we ignore padding tokens. when it is 1, it means we ignore <unk> or oov words
    Returns
    -------

    """
    assert query.size(0) == doc.size(0)
    assert len(query.size()) == 2 and len(doc.size()) == 2
    query_mask = query > threshold
    doc_mask = doc > threshold
    query_mask = query_mask.unsqueeze(2)  # (B, L, 1)
    doc_mask = doc_mask.unsqueeze(2)  # (B, R, 1)
    doc_mask = doc_mask.permute(0, 2, 1)  # (B, 1, R)

    mask_tensor = torch.bmm(query_mask.float(), doc_mask.float())  # (B, L, R)
    return mask_tensor  # , torch.sum(query_mask, dim = 1).squeeze(), torch.sum(doc_mask, dim = 1).squeeze()


def create_mask_tensor_image(left_indices: torch.Tensor, right_indices: torch.Tensor, threshold: int = 0):
    """
    Creating masking of two tensor. These two tensors are integer tensor
    Parameters

    ----------
    left_indices: (B1, n1, M1)
    right_indices: (B, n, M2)
    threshold: when it is 0, means we ignore padding tokens. when it is 1, it means we ignore <unk> or oov words
    Returns
    -------

    """
    B1, n1, M1 = left_indices.size()
    B, n, M2 = right_indices.size()
    assert n1 == 1
    left_mask = left_indices > 0
    right_mask = right_indices > 0
    left_mask = left_mask.view(B1, M1, 1)
    if B1 == 1: left_mask = left_mask.expand(B, M1, 1)  # during testing
    right_mask = right_mask.view(B, n * M2, 1)
    ans = torch.bmm(left_mask.float(), right_mask.permute(0, 2, 1).float())
    ans = ans.view(B, M1, n, M2).permute(0, 2, 1, 3)  # (B, n, M1, M2)
    return ans


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_sorted_index_and_reverse_index(base_array: np.ndarray):
    """
    We use sorted_index = np.argsort(-base_array) to find the indices to short the array decreasingly.
    We also need to find the indices to restoring the original order of elements of base_array
    after apply sorted_index.
    This method is important because we need to input the tensor to GRU/LSTM with packed sequence.
    Parameters
    ----------
    base_array: (B, )

    Returns
    -------

    """
    assert type(base_array) == np.ndarray
    batch_size = base_array.shape[0]
    assert base_array.shape == (batch_size,)
    new_indices = np.argsort(-base_array)
    old_indices = np.arange(batch_size)
    r = np.stack([new_indices, old_indices], axis = 1)
    r = r[np.argsort(r[:, 0])]
    restoring_indices = r[:, 1]  # the retoring indices. This method is tested very carefully.
    return new_indices, restoring_indices


def packing_sequence(seq: torch.Tensor, seq_lens: np.ndarray, new_index) -> torch.Tensor:
    """
    Prepare a packed sequence to input to an RNN. It is required that the length of sequences in `seq` must be sorted.
    After

    Parameters
    ----------
    seq: (B, L, D) where L is length of sequence
    seq_lens: (B, )
    new_index: (B, ) this index is used to make sequence lengths sorted
    old_index: (B, ) this index is used to restore the sequence lengths
    Returns
    -------

    """
    return rnn_utils.pack_padded_sequence(seq[new_index], seq_lens[new_index], batch_first = True)


def torch_repeat_dim0(A: torch.tensor, n: int):
    """
    Repeat tensor across a dimension
    Parameters
    ----------
    A
    axis

    Returns
    -------

    """
    assert len(A.size()) == 3
    d1, d2, d3 = A.size()
    A = A.unsqueeze(0).transpose(0, 1).repeat(1, n, 1, 1).view(-1, d2, d3)
    assert A.size() == (n * d1, d2, d3)
    return A


def boolean_mask(target: torch.Tensor, mask: torch.Tensor):
    """
    Mimick tf.boolean_mask
    Copied from https://discuss.pytorch.org/t/slicing-tensor-using-boolean-list/7354/3
    Parameters
    ----------
    target
    mask

    Returns
    -------

    """
    x = mask == True
    # y=torch.arange(0,3)
    # x=torch.Tensor([True,False,True])==True
    # print(y[x])
    return target[x]

def torch_argsort(input, dim=None, descending=False):
    """Returns the indices that sort a tensor along a given dimension in ascending
        order by value.
         This is the second value returned by :meth:`torch.sort`.  See its documentation
        for the exact semantics of this method.
         Args:
            input (Tensor): the input tensor
            dim (int, optional): the dimension to sort along
            descending (bool, optional): controls the sorting order (ascending or descending)
         Example::
             >>> a = torch.randn(4, 4)
            >>> a
            tensor([[ 0.0785,  1.5267, -0.8521,  0.4065],
                    [ 0.1598,  0.0788, -0.0745, -1.2700],
                    [ 1.2208,  1.0722, -0.7064,  1.2564],
                    [ 0.0669, -0.2318, -0.8229, -0.9280]])
             >>> torch.argsort(a, dim=1)
            tensor([[2, 0, 3, 1],
                    [3, 2, 1, 0],
                    [2, 1, 0, 3],
                    [3, 2, 1, 0]])
    """
    # copy from https://github.com/pytorch/pytorch/pull/9600/files
    if dim is None:
        return torch.sort(input, -1, descending)[1]
    return torch.sort(input, dim, descending)[1]


def _predict_process_ids(user_ids, item_ids, num_items, use_cuda):
    """

    Parameters
    ----------
    user_ids
    item_ids
    num_items
    use_cuda

    Returns
    -------

    """
    if item_ids is None:
        item_ids = np.arange(num_items, dtype=np.int64)

    if np.isscalar(user_ids):
        user_ids = np.array(user_ids, dtype=np.int64)

    user_ids = torch.from_numpy(user_ids.reshape(-1, 1).astype(np.int64))
    item_ids = torch.from_numpy(item_ids.reshape(-1, 1).astype(np.int64))
    if item_ids.size()[0] != user_ids.size(0):
        user_ids = user_ids.expand(item_ids.size())

    user_var = gpu(user_ids, use_cuda)
    item_var = gpu(item_ids, use_cuda)

    return user_var.squeeze(), item_var.squeeze()


def idf(total_docs: int, term_freq: int) -> float:
    """compute inverse doc frequency. If a term appears at all docs, then, its value is low for discrimination.
    If a term does not show in any doc, then, we simply use set denominator to 1 => largest idf value """
    assert term_freq <= total_docs, "The number of documents that contain a term must be smaller than total_docs"
    return np.log((1.0 + total_docs) / float(term_freq + 1.0)) + 1.0


def moving_average(input_tensor: torch.Tensor, window_size: int, dimension: int):
    """

    Parameters
    ----------
    input_tensor: torch.Tensor  of shape (B, L, D)
    window_size: sliding windows size
    dimension: dimension we want to apply sliding window

    Returns
    -------

    """
    ret = torch.cumsum(input_tensor, dim = dimension)
    # print("Here:", ret, ret.shape)
    ret[:, window_size:] = ret[:, window_size:] - ret[:, :-window_size]
    return ret[:, window_size - 1:] / window_size


def cosine_distance(a: torch.Tensor, b: torch.Tensor):
    """
    Compute the cosine distance between two tensors. This implementation saves a lot of memory since
    memory complexity is O(B x L x R)
    Parameters
    ----------
    a: `torch.Tensor` shape (B, L, D)
    b: `torch.Tensor` shape (B, R, D)

    Returns
    -------

    """
    assert len(a.size()) == len(b.size()) == 3
    A_square = (a * a).sum(dim = - 1)  # B, L
    B_square = (b * b).sum(dim = -1)  # B, R
    dot = torch.bmm(a, b.permute(0, 2, 1))  # B, L, R
    # added abs in case of negative, added 1e-10 to avoid nan gradient of sqrt
    return torch.sqrt(torch.abs(A_square.unsqueeze(-1) - 2 * dot + B_square.unsqueeze(1)) + 1e-10)


def l1_distance(a: torch.Tensor, b: torch.Tensor):
    """
    Compute the l1 distance between two tensors. This implementation consumes a lot of memory since
    mem complexity is O(B x L x R x D) due to x - y. I tried many ways but this is the best thing I can do
    Parameters
    ----------
    a: `torch.Tensor` shape (B, L, D)
    b: `torch.Tensor` shape (B, R, D)

    Returns
    -------

    """
    assert len(a.size()) == len(b.size()) == 3
    x = a.unsqueeze(2)  # (B, L, 1, D)
    y = b.unsqueeze(1)  # (B, 1, R, D)
    return torch.norm(x - y, p = 1, dim = -1)


def _get_doc_context_copacrr(doc: torch.Tensor, doc_mask: torch.Tensor, context_window_size: int) -> torch.Tensor:
    """

    Parameters
    ----------
    doc: with shape (B, R, D)
    doc_mask: binary tensor that differentiate real tokens from padding tokens (B, R)

    Returns
    -------
    a tensor of shape (B, R, D) which indicates the context representation of each token in doc.
    We also reset padding tokens to zero since they have no context
    """

    def moving_average(a: torch.Tensor, window_size: int, dimension: int):
        ret = torch.cumsum(a, dim = dimension)
        # print("Here:", ret, ret.shape)
        ret[:, window_size:] = ret[:, window_size:] - ret[:, :-window_size]
        return ret[:, window_size - 1:] / window_size

    left = context_window_size // 2
    right = context_window_size - left - 1  # in case context windows is an even number then left=x//2, right=x-x//2
    y = F.pad(doc, (0, 0, left, right))  # (B, c/2 + R + c/2, D)
    document_context = moving_average(y, window_size = context_window_size, dimension = 1)
    document_context = document_context * doc_mask.unsqueeze(-1).float()
    return document_context


def init_weights(m):
    """
    Copied from https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/3
    Examples:
        >>> w = nn.Linear(3, 4)
        >>> w.apply(init_weights)
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m.bias, "data"): m.bias.data.fill_(0)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias:
            torch.nn.init.xavier_uniform_(m.bias)


def auto_rnn(rnn_cell: nn.RNN, input_feats: torch.Tensor,
             lens: torch.Tensor, new_indices: torch.Tensor, restoring_indices: torch.Tensor, max_len: int):
    """

    Parameters
    ----------
    rnn_cell : a rnn cell
    input_feats: `torch.Tensor` (B, L, D)
    lens: `torch.Tensor` (B, )
    new_indices: `torch.Tensor` (B, )
    restoring_indices: `torch.Tensor` (B, )
    max_len: int
    Returns
    -------

    """
    return rnn_cell((input_feats, lens, new_indices, restoring_indices), max_len=max_len, return_h=False)[0]


def rnn_last_h(rnn_cell: nn.RNN, input_feats: torch.Tensor,
               lens: torch.Tensor, new_indices: torch.Tensor, restoring_indices: torch.Tensor, max_len: int):
    """
    return the last hidden vectors of an RNN
    Parameters
    ----------
    rnn_cell : a rnn cell
    input_feats: `torch.Tensor` (B, L, D)
    lens: `torch.Tensor` (B, )
    new_indices: `torch.Tensor` (B, )
    restoring_indices: `torch.Tensor` (B, )
    max_len: int
    Returns
    -------

    """
    return rnn_cell((input_feats, lens, new_indices, restoring_indices), max_len=max_len, return_h=True)[1]


def retrieve_elements_from_indices(tensor: torch.Tensor, indices: torch.Tensor):
    """
    Copied from https://discuss.pytorch.org/t/pooling-using-idices-from-another-max-pooling/37209/4
    How does this work? (Checked
    Parameters
    ----------
    tensor: torch.Tensor shape B, C, L, R
    indices: torch.Tensor shape (B, C, L, R) the values are indices where the last two dimensions are flattened

    Returns
    -------

    """
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    return output


data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_images(infile):
    im = Image.open(infile).convert('RGB')
    return data_transforms(im)
