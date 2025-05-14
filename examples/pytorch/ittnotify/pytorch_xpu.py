import torch
import intel_extension_for_pytorch as ipex
device = torch.device('xpu')

torch.manual_seed(0)

src = torch.rand((2048, 1, 512))
tgt = torch.rand((2048, 20, 512))
dataset = torch.utils.data.TensorDataset(src, tgt)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

model = torch.nn.Transformer(batch_first=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
model.train()
model = model.to(device)
criterion = criterion.to(device)
model, optimizer = ipex.optimize(model, optimizer=optimizer)

with torch.autograd.profiler.emit_itt():
    for epoch in range(10):
        print(f'Epoch {epoch}')
        for source, targets in loader:
            source = source.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            output = model(source, targets)
            loss = criterion(output, targets)

            loss.backward()
            optimizer.step()
