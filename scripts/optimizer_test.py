import torch
from torch import nn, optim


IN_DIM = 2
OUT_DIM = 3
BATCH_SIZE = 4
STEPS = 10


class GrowModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.mods = nn.ModuleList([nn.Linear(IN_DIM, OUT_DIM)])

    def grow(self):
        self.mods.append(nn.Linear(IN_DIM, OUT_DIM))

    def forward(self, x):
        return sum(mod(x) for mod in self.mods)


network = GrowModule()
optimizer = optim.Adam(network.parameters(), lr=3e-4, eps=1e-8)

for step in range(STEPS):

    batch = torch.rand((BATCH_SIZE, IN_DIM))
    out = network(batch)

    network.zero_grad()
    loss = torch.sum(out ** 2)
    loss.backward()
    optimizer.step()

    if step == STEPS // 2:
        print(optimizer.state_dict())
        network.grow()
        optimizer.add_param_group({"params": network.mods[-1].parameters()})

print("")
print(optimizer.state_dict())
