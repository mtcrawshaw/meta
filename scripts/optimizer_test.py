from torch import nn, optim


class GrowModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.mods = nn.ModuleList([nn.Linear(2, 3)])

    def grow(self):
        self.mods.append(nn.Linear(3, 4))


network = GrowModule()
optimizer = optim.Adam(network.parameters(), lr=3e-4, eps=1e-8)
print("before:")
print(optimizer.param_groups)

network.grow()
print("\nafter:")
print(optimizer.param_groups)

optimizer.param_groups.append({"params": network.mods[-1].parameters()})
print("\nreally after:")
print(optimizer.param_groups)
