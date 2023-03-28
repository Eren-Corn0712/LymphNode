import json
import matplotlib.pyplot as plt

file = "resnet50/1-fold/log.txt"


log_data = []
with open(file, "r") as f:
    for line in f:
        log_data.append(json.loads(line))

train_loss = []
train_global_loss = []
epochs = []

for log in log_data:
    train_loss.append(log['train_loss'])
    train_global_loss.append(log['train_global_loss'])

    epochs.append(log['epoch'])

plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, train_global_loss, label='Train Global Loss')

plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.show()