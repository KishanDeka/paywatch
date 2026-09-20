import torch
from paywatch.lstm import LSTMCell, LSTMAutoencoder


def test_cell_matches_pytorch_forward_and_gradients():
    torch.manual_seed(5)
    custom = LSTMCell(3, 4).double()
    reference = torch.nn.LSTMCell(3, 4).double()
    with torch.no_grad():
        reference.weight_ih.copy_(custom.weight_ih)
        reference.weight_hh.copy_(custom.weight_hh)
        reference.bias_ih.copy_(custom.bias)
        reference.bias_hh.zero_()
    x = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    h, c = torch.randn(2, 4, dtype=torch.double), torch.randn(2, 4, dtype=torch.double)
    a = custom(x, (h, c))
    b = reference(x, (h, c))
    for actual, expected in zip(a, b):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    sum(v.sum() for v in a).backward(retain_graph=True)
    gradient = x.grad.clone()
    x.grad.zero_()
    sum(v.sum() for v in b).backward()
    torch.testing.assert_close(x.grad, gradient)
    torch.testing.assert_close(custom.weight_ih.grad, reference.weight_ih.grad)
    torch.testing.assert_close(custom.weight_hh.grad, reference.weight_hh.grad)
    torch.testing.assert_close(custom.bias.grad, reference.bias_ih.grad)


def test_cell_numerical_gradient():
    cell = LSTMCell(2, 2).double()
    x, h, c = [torch.randn(1, 2, dtype=torch.double, requires_grad=True) for _ in range(3)]
    assert torch.autograd.gradcheck(lambda x, h, c: cell(x, (h, c)), (x, h, c))


def test_saturated_inputs_remain_finite():
    cell = LSTMCell(3, 4)
    h, c = cell(torch.full((2, 3), 1e6), (torch.zeros(2, 4), torch.zeros(2, 4)))
    assert torch.isfinite(h).all() and torch.isfinite(c).all()


def test_autoencoder_learns_small_normal_batch():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = LSTMAutoencoder(features=3, hidden=8, length=4)
    x = torch.randn(12, 4, 3) * 0.05 + 0.5
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    initial = (model(x) - x).square().mean().item()
    for _ in range(35):
        optimizer.zero_grad()
        loss = (model(x) - x).square().mean()
        loss.backward()
        optimizer.step()
    assert (model(x) - x).square().mean().item() < initial * 0.2
