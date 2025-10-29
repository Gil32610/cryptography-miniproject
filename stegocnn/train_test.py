import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import time as tm
import datetime
import os
from sklearn.metrics import accuracy_score
from torch.amp import autocast, GradScaler


scaler = GradScaler(device="cuda", enabled=True)

def evaluate_model(model, data_loader, loss_fn, device):
    """Evaluates the model on a given dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    total_batches = len(data_loader)
    
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(data_loader, start=1):
            print(f"Batch {batch_idx}/{total_batches}", end="\r")
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = loss_fn(outputs, y)
            print(loss)
            total_loss += loss.item() * X.size(0)

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    print()
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    torch.cuda.empty_cache()
    return avg_loss, accuracy

def train_pytorch(model,dataset_train, dataset_val, dataset_test, model_name="", path_log_base="logs", batch_size=32, epochs=100):
    """
    Trains a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to train.
        datasets_* : torch Dataset objects
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        model_name (str): Name for logging and saving.
        path_log_base (str): Base directory for logs and checkpoints.
    """
    
    start_time = tm.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    train_loader = DataLoader(dataset_train, collate_fn=pair_collate_fn, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset_val, collate_fn=pair_collate_fn, batch_size=batch_size)
    test_loader  = DataLoader(dataset_test, collate_fn=pair_collate_fn, batch_size=batch_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    log_timestamp = datetime.datetime.now().isoformat()[:19].replace("T", "_").replace(":", "-")
    log_dir = os.path.join(path_log_base, f"{model_name}_{log_timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    

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
            
            optimizer.zero_grad()
            
            with autocast('cuda',dtype=torch.float16):
                outputs = model(X)
                loss = loss_fn(outputs, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        
        val_loss, val_accuracy = evaluate_model(model, valid_loader, loss_fn, device)
        torch.cuda.empty_cache()
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

def train_knowledge_distillation(teacher, student, dataset_train, dataset_val, dataset_test, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device, model_name="", path_log_base="logs", batch_size=32):
    start_time = tm.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student.to(device)

    train_loader = DataLoader(dataset_train, collate_fn=pair_collate_fn, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset_val, collate_fn=pair_collate_fn, batch_size=batch_size)
    test_loader  = DataLoader(dataset_test, collate_fn=pair_collate_fn, batch_size=batch_size)

    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    log_timestamp = datetime.datetime.now().isoformat()[:19].replace("T", "_").replace(":", "-")
    log_dir = os.path.join(path_log_base, f"{model_name}_{log_timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    lossTEST, accuracyTEST = evaluate_model(student, test_loader, ce_loss, device)
    lossTRAIN, accuracyTRAIN = evaluate_model(student, train_loader, ce_loss, device)
    lossVALID, accuracyVALID = evaluate_model(student, valid_loader, ce_loss, device)

    teacher.eval()  # Teacher set to evaluation mode

    print(f"Initial Metrics - Train Loss: {lossTRAIN:.4f}, Acc: {accuracyTRAIN:.4f} | Valid Loss: {lossVALID:.4f}, Acc: {accuracyVALID:.4f} | Test Loss: {lossTEST:.4f}, Acc: {accuracyTEST:.4f}")
    print("Starting the training...")
    for epoch in range(epochs):
        student.train() # Student to train mode
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(inputs)

            student_logits = student(inputs)

            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1) # maybe 1? João
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1) # maybe 1? João

            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

            label_loss = ce_loss(student_logits, labels)

            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        
        val_loss, val_accuracy = evaluate_model(student, valid_loader, ce_loss, device)
        
        print(f"Epoch {epoch:03d}/{epochs} - Loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', val_loss, epoch)
        writer.add_scalar('Accuracy/valid', val_accuracy, epoch)
        
        checkpoint_path = os.path.join(log_dir, f"saved-model-{epoch:03d}-{val_accuracy:.4f}.pth")
        torch.save(student.state_dict(), checkpoint_path)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")
    
    writer.close()
    
    final_test_loss, final_test_accuracy = evaluate_model(student, test_loader, ce_loss, device)
    
    TIME = tm.time() - start_time
    print(f"Time {model_name} = {TIME:.2f} [seconds]")
    print("\n")
    print(log_dir)
    
    return {"test_loss": final_test_loss, "test_accuracy": final_test_accuracy}
def pair_collate_fn(batch):
    images, labels = [], []
    
    for cover_pair, stego_pair in batch:
        cover_img, cover_lbl = cover_pair
        stego_img, stego_lbl = stego_pair
        images.extend([cover_img, stego_img])
        labels.extend([cover_lbl, stego_lbl])
    
    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels

