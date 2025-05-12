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
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Define all tunable hyperparameters in one place
# Define hyperparameters in a dictionary
# hyperparams = {
#     "learning_rate": 0.001,
#     "batch_size": 32,
#     "epochs": 20,
#     "dropout_rate": 0.5
# }

def train_model(model, train_loader, val_loader, device, epochs=40, lr=1e-4):
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
def plot_metrics(train_acc, val_acc, train_loss, val_loss, type, acc):
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'train_loss_plot_{type}_{acc:.2g}.png')  # Save as .png (or .jpg/.pdf if you want)
    plt.close()  # Close the figure so it doesn't stay in memory or display

def hyper_tuning_VGG():
    # Hyperparameter options
    # batch_sizes = [32]
    # learning_rates = [1e-3, 1e-4]
    batch_size=32
    learning_rates=[0.001,0.0001]
    dropout_rates = [0.3,0.5,0.6]

    lr=0.001
    dropout=0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_val_acc = 0.89
    best_model_state = None
    best_config = {}

    for lr, dropout in itertools.product( learning_rates, dropout_rates):
    # for i in range(0,10):
        print(f"\nTesting: batch_size={batch_size}, lr={lr}, dropout={dropout}")

        train_loader, val_loader, test_loader, class_names = pre.data_preprocess(batch_size=batch_size)
        model = m.VGG9( dropout_rate=dropout).to(device)

        train_acc, val_acc, train_loss, val_loss = train_model(model, train_loader, val_loader, device, lr=lr)

        final_val_acc = val_acc[-1]
        if final_val_acc > best_val_acc:
            best_val_acc = final_val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_config = {
                'batch_size': batch_size,
                'epochs': 30,
                'learning_rate': lr,
                'dropout_rate': dropout
            }
            # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_metrics(train_acc, val_acc, train_loss, val_loss,"VGG", best_val_acc)
            torch.save(best_model_state, f'best_vgg9_model_{final_val_acc:.3g}.pt')
            # torch.save(best_model_state, f'best_vgg9_model_{lr}_{dropout}.pt')
            print(f"Best model so far saved with Val Acc: {best_val_acc:.4f}")

    # Report best result
    print("\n Best Hyperparameter Configuration:")
    for k, v in best_config.items():
        print(f"{k}: {v}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    return best_config
    

def hyper_tuning_resnet(learning_rates, dropout_rates, epochs=30):
    pre.top8classes_finding()

    best_val_acc = 0.00
    best_model = None
    best_config = None


    train_loader, val_loader, test_loader, class_names = pre.data_preprocess(batch_size=32)

    for lr in learning_rates:
        for dr in dropout_rates:
            print(f"\nTraining with lr={lr}, dropout={dr}")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = m.CustomResNet18(dropout_rate=dr).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            train_acc, val_acc, train_loss, val_loss = train_model(model, train_loader, val_loader, device, epochs)

            if val_acc[-1] > best_val_acc:
                best_val_acc = val_acc[-1]
                best_model = model
                best_config = (lr, dr)
                plot_metrics(train_acc, val_acc, train_loss, val_loss, "resnet", best_val_acc)
                torch.save(best_model.state_dict(), f"best_resnet18_model_{best_val_acc:.3g}.pt")

    print(f"\nBest config: lr={best_config[0]}, dropout={best_config[1]}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    return best_config

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print classification report
    print("Classification Report for each classes:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print("=== Overall Performance on Test Set ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # cm = confusion_matrix(all_labels, all_preds)
    # plt.figure(figsize=(10,8))
    # sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.show()


if __name__ == '__main__':
    # pre.top8classes_finding()
    train_loader, val_loader, test_loader, class_names = pre.data_preprocess() # batch is 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Call the function
    # model = m.VGG9(num_classes=8).to(device)  # make sure num_classes matches your dataset
    # model.load_state_dict(torch.load("best_vgg9_model_0.921.pt", map_location=device,weights_only=True))
    # model.eval()
    # evaluate_model(model, test_loader, device, class_names)

    model=m.CustomResNet18().to(device)
    model.load_state_dict(torch.load("best_resnet18_model_0.98.pt", map_location=device,weights_only=True))
    model.eval()
    evaluate_model(model, test_loader, device, class_names)
   
    # model=m.VGG11().to(device)
    # train_acc_list, val_acc_list, train_loss_list, val_loss_list=train_model(model, train_loader, val_loader, device)
    # plot_metrics(train_acc_list, val_acc_list)

    # learning_rates=[1e-3]
    # dropout_rates = [0.6,0.6,0.6,0.6,0.6]
    # best_config=hyper_tuning_resnet(learning_rates, dropout_rates)
    # resnet 0.001lr, dp0.5

    # best_config=hyper_tuning_VGG()





