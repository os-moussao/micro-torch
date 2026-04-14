"""
This is exactly Test 04 but with Adam Optimizer
"""

import pandas as pd
import numpy as np

data = pd.read_csv('./data.csv', header=None)

data = data.drop(columns=0)

X = data.iloc[:,1:].values
y = pd.get_dummies(data[1], dtype=int).values

from microtorch.data import random_split, StandardScaler, DataLoader

(X_train, y_train), (X_test, y_test) = random_split(X, y, ratios=[0.8, 0.2], seed=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

BATCH_SIZE=16

train_loader = DataLoader(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
)

test_loader = DataLoader(
    X_test, y_test,
    batch_size=BATCH_SIZE
)


LR = 0.03
epochs = 5


from microtorch import models
from microtorch.optimizers import Adam
from microtorch.functions import cross_entropy_loss
from microtorch import nn
from tqdm import tqdm

np.random.seed(1337)

n_features = X.shape[1]

model = models.MLP(
    in_size=n_features,
    out_size=2,
    hidden_layers=[10],
    activations=['relu']
)

softmax = nn.Softmax()

optimizer = Adam(model, lr=LR)

loss_fn = cross_entropy_loss

def acc_fn(y_pred, y_true):
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    return (y_pred == y_true).mean()


loss_hist = []
train_acc_h = []
test_acc_h = []

train_acc = acc_fn(model(X_train), y_train)
test_acc = acc_fn(model(X_test), y_test)

print(f"Before Training: Train Accuracy {train_acc:.2f} | Test Accuracy {test_acc:.2f}")

for e in range(epochs):
    train_loss = 0
    for X, y in tqdm(train_loader, f"Epoch #{e}"):
        optimizer.zero_grad()
        
        y_pred = model(X)
        
        y_pred = softmax(y_pred)
    
        l = loss_fn(y_pred, y)

        train_loss += l.val

        l.backprop()
        
        optimizer.step()

    train_loss /= len(train_loader)

    # accuracy
    train_acc = acc_fn(model(X_train), y_train)
    test_acc = acc_fn(model(X_test), y_test)

    loss_hist.append(train_loss)
    train_acc_h.append(train_acc)
    test_acc_h.append(test_acc)

    print(f"Epoch #{e}: Loss {train_loss} | Train Accuracy {train_acc:.2f} | Test Accuracy {test_acc:.2f}")