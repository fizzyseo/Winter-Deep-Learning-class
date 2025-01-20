# %%
# Package 준비
import torch
from torch import nn
from torch import optim

# %%
# 데이터 준비
a = 0.1
b = 0.3

x = torch.normal(0.0, 0.55, (10000, 1))
y = a * x + b + torch.normal(0.0, 0.03, (10000, 1))

# %%
# 딥러닝 모델 생성
param = torch.randn(2, requires_grad=True)

# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.linear = nn.Linear(1, 1)

#     def forward(self, X):
#         X = self.linear(X)
#         return X

# model = nn.Linear(1, 1) # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
# model = Model()

# param = list(model.parameters())     
print(f"Initial parameters")
print(f"a: {param[0].item():.3f}, b: {param[1].item():.3f}\n")

# %%
# 학습 방법 준비
## 오차함수 
## Greaidnet descent 알고리즘

LEARNING_RATE = 0.05

criterion  = nn.MSELoss()
optimizer = optim.SGD([param], lr=LEARNING_RATE)
# optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# %%
# Training loop
print("Start training")
EPOCHS = 200
for epoch in range(EPOCHS):
    y_pred = param[0]*x + param[1]
    # y_pred = model.forward(x)

    loss = criterion(y_pred, y)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if (epoch == 0) or ((epoch+1) % 10 == 0):
        # param = list(model.parameters())        
        y_pred = param[0].detach() * x + param[1].detach()
        print(f"Epoch: {epoch+1}, Loss: {loss.data.numpy():.4f}, a: {param[0].item():.3f}, b: {param[1].item():.3f}")
