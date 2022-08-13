#%% 1. 케라스 패키지 임포트
import torch

#%% 2. 데이터 지정
x = torch.tensor([0., 1., 2., 3., 4.]).view(-1, 1)
y = x * 2 + 1

#%% 3. 인공신경망 모델링
model = torch.nn.Sequential( torch.nn.Linear(1, 1))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#%% 4. 주어진 데이터로 모델 학습
def fit(optimizer, x_a, y_a, epochs=1000):
    for epoch in range(epochs):
        y_pred = model(x_a)
        loss = (y_pred - y_a).pow(2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

fit(optimizer, x[:2], y[:2], epochs=1000)

#%% 5. 성능 평가
y_pred = model(x[2:]).flatten()
print('Targets:', y[2:])
print('Predictions:', y_pred)
print('Errors:', y[2:] - y_pred)
# %%
