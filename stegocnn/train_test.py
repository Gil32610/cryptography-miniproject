import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import time as tm
import datetime
import os
from sklearn.metrics import accuracy_score


def evaluate_model(model, data_loader, loss_fn, device):
    """Evaluates the model on a given dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = loss_fn(outputs, y)
            total_loss += loss.item() * X.size(0)

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

def train_pytorch(model, X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size, epochs, model_name="", path_log_base="logs"):
    """
    Trains a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to train.
        X_train, y_train, X_valid, y_valid, X_test, y_test (np.ndarray or torch.Tensor): Data splits.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        model_name (str): Name for logging and saving.
        path_log_base (str): Base directory for logs and checkpoints.
    """
    
    start_time = tm.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long) # Use torch.long for classification labels
    X_valid_t = torch.tensor(X_valid, dtype=torch.float32)
    y_valid_t = torch.tensor(y_valid, dtype=torch.long)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    valid_dataset = TensorDataset(X_valid_t, y_valid_t)
    test_dataset  = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    log_timestamp = datetime.datetime.now().isoformat()[:19].replace("T", "_").replace(":", "-")
    log_dir = os.path.join(path_log_base, f"{model_name}_{log_timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    best_val_accuracy = -1.0 # For saving the best model

    lossTEST, accuracyTEST = evaluate_model(model, test_loader, loss_fn, device)
    lossTRAIN, accuracyTRAIN = evaluate_model(model, train_loader, loss_fn, device)
    lossVALID, accuracyVALID = evaluate_model(model, valid_loader, loss_fn, device)

    print(f"Initial Metrics - Train Loss: {lossTRAIN:.4f}, Acc: {accuracyTRAIN:.4f} | Valid Loss: {lossVALID:.4f}, Acc: {accuracyVALID:.4f} | Test Loss: {lossTEST:.4f}, Acc: {accuracyTEST:.4f}")
    print("Starting the training...")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            
            outputs = model(X)
            loss = loss_fn(outputs, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        
        val_loss, val_accuracy = evaluate_model(model, valid_loader, loss_fn, device)
        
        print(f"Epoch {epoch:03d}/{epochs} - Loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', val_loss, epoch)
        writer.add_scalar('Accuracy/valid', val_accuracy, epoch)
        
        checkpoint_path = os.path.join(log_dir, f"saved-model-{epoch:03d}-{val_accuracy:.4f}.pth")
        torch.save(model.state_dict(), checkpoint_path)

    writer.close()
    
    final_test_loss, final_test_accuracy = evaluate_model(model, test_loader, loss_fn, device)
     
    TIME = tm.time() - start_time
    print(f"Time {model_name} = {TIME:.2f} [seconds]")
    
    print("\n")
    print(log_dir)
    
    return {"test_loss": final_test_loss, "test_accuracy": final_test_accuracy}

def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Forward pass with the student model
            student_logits = student(inputs)

            #Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")