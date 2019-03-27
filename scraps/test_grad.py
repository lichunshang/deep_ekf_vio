import torch

# x = torch.tensor(([1.]), requires_grad=True)
# y = x ** 2
# z = 2 * y
# w = z ** 3

x = torch.tensor(([1.]), requires_grad=True)
x2 = x.clone()
x2[0] = x ** 2
z = 2 * x2[0]
w = z ** 3

w.backward()
print(x.grad)
