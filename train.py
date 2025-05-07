import data_pre as pre
import model as m
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import itertools
from datetime import datetime
import copy

# Define all tunable hyperparameters in one place
# Define hyperparameters in a dictionary
# hyperparams = {
#     "learning_rate": 0.001,
#     "batch_size": 32,
#     "epochs": 20,
#     "dropout_rate": 0.5
# }

def train_model(model, train_loader, val_loader, device, epochs=30, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_acc_list, val_acc_list = [], []
    train_loss_list, val_loss_list = [], []

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss / len(val_loader))

        print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    return train_acc_list, val_acc_list, train_loss_list, val_loss_list

# Plotting
def plot_metrics(train_acc, val_acc):
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def hyper_tuning():
    # Hyperparameter options
    # batch_sizes = [32]
    # learning_rates = [1e-3, 1e-4]
    batch_sizes=[32]
    epochs_list= [30]
    learning_rates=[0.0001]
    dropout_rates = [0.3, 0.4, 0.5, 0.6, 0.7]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_val_acc = 0
    best_model_state = None
    best_config = {}

    for batch_size, lr, dropout in itertools.product(batch_sizes, learning_rates, dropout_rates):
        print(f"\nTesting: batch_size={batch_size}, lr={lr}, dropout={dropout}")

        train_loader, val_loader, class_names = pre.data_preprocess(batch_size=batch_size)
        model = m.VGG11(num_classes=len(class_names), dropout_rate=dropout).to(device)

        train_acc, val_acc, train_loss, val_loss = train_model(model, train_loader, val_loader, device, lr=lr)

        final_val_acc = val_acc[-1]
        if final_val_acc > best_val_acc:
            best_val_acc = final_val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_config = {
                'batch_size': batch_size,
                'epochs': 20,
                'learning_rate': lr,
                'dropout_rate': dropout
            }
            # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # torch.save(best_model_state, f'best_vgg11_model_{timestamp}.pt')
            print(f"Best model so far saved with Val Acc: {best_val_acc:.4f}")

    # Report best result
    print("\n Best Hyperparameter Configuration:")
    for k, v in best_config.items():
        print(f"{k}: {v}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    return best_config
    

def hyper_tuning_resnet(batch_sizes, learning_rates, dropout_rates, epochs=20):
    pre.top8classes_finding()

    best_val_acc = 0.0
    best_model = None
    best_config = None


    train_loader, val_loader, class_names = pre.data_preprocess(batch_size=32)

    for lr in learning_rates:
        for dr in dropout_rates:
            print(f"\nTraining with lr={lr}, dropout={dr}")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = m.get_custom_resnet18(num_classes=len(class_names), dropout_rate=dr).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            train_acc, val_acc, _, _ = train_model(model, train_loader, val_loader, device, epochs)

            if val_acc[-1] > best_val_acc:
                best_val_acc = val_acc[-1]
                best_model = model
                best_config = (lr, dr)
                torch.save(best_model.state_dict(), "best_resnet50_model.pth")

    print(f"\nBest config: batch_size={best_config[0]}, lr={best_config[1]}, dropout={best_config[2]}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    return best_config


if __name__ == '__main__':
    pre.top8classes_finding()
    learning_rates=[1e-3, 1e-4]
    dropout_rates = [ 0.3, 0.5, 0.7]
    best_config=hyper_tuning_resnet(learning_rates, dropout_rates)





