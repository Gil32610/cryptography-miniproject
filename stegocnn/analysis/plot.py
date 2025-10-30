import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from train_test import pair_collate_fn
from captum.attr import IntegratedGradients
from sklearn.metrics import ( 
                             roc_curve, 
                             auc, 
                             confusion_matrix, 
                             accuracy_score, 
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score
                             )

class Plotter:
    def __init__(self, dataset):
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.y_true_list = []
        self.y_pred_list = []
        self.y_proba_list = []
        
    
    def get_metrics_results(self, model:torch.nn.Module, batch_size=16, num_works=1):
        loader  = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            num_workers=num_works,
            shuffle=False,
            collate_fn=pair_collate_fn
            )
        model.to(self.device)
        model.eval()
        
        y_true_list = []
        y_pred_list = []
        y_proba_list = []
        
        print(f"Calculating metrics for {model}:") 
        
        with torch.no_grad():
            for batch_index, (X, y) in enumerate(loader):
                X, y = X.to(self.device), y.to(self.device)
                outputs = model(X)
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(1)
                
                probs = torch.softmax(outputs,dim=1)
                _, preds = torch.max(probs,1)
                y_true_list.append(y.detach().cpu().numpy())
                y_pred_list.append(preds.detach().cpu().numpy())
                y_proba_list.append(probs.detach().cpu().numpy())
        self.y_true = np.concatenate(y_true_list, axis=0).ravel()
        self.y_pred = np.concatenate(y_pred_list, axis=0).ravel()
        self.y_proba = np.concatenate(y_proba_list, axis=0)
        
        if self.y_proba.ndim == 1:
            self.y_proba = self.y_proba.reshape(-1, 1)
            
        return {
            "accuracy": accuracy_score(self.y_true, self.y_pred),
            "precision": precision_score(self.y_true, self.y_pred, average='binary' if self.y_proba.shape[1] == 2 else 'macro', zero_division=0),
            "recall": recall_score(self.y_true, self.y_pred, average='binary' if self.y_proba.shape[1] == 2 else 'macro', zero_division=0),
            "f1": f1_score(self.y_true, self.y_pred, average='binary' if self.y_proba.shape[1] == 2 else 'macro', zero_division=0),
        }
                

    def plot_roc_curve(self, save_path=None, model_name=None):
        # Ensure probabilities are available
        if not hasattr(self, "y_proba"):
            raise ValueError("Run get_metrics_results() first to populate probabilities.")

        y_true = self.y_true
        y_proba = self.y_proba

        # Handle binary classification (2 columns: class 0 and class 1)
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            y_score = y_proba[:, 1]  # probability for the positive class (Stego)
        elif y_proba.ndim == 2 and y_proba.shape[1] > 2:
            raise ValueError(
                f"ROC curve is not well-defined for {y_proba.shape[1]} classes; use one-vs-rest approach."
            )
        else:
            # Single output case
            y_score = y_proba.ravel()

        # Compute ROC and AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)

        # Plot
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='blue', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve {f"– {model_name}" if model_name else ""}')
        plt.legend(loc="lower right")

        # Save or show
        if save_path:
            plt.savefig(f"{save_path}/roc_curve_{model_name or 'model'}.png",
                        bbox_inches="tight")
        plt.close()
        
    def plot_confusion_matrix(self, save_path=None, model_name=None):
        if not hasattr(self, "y_true") or not hasattr(self, "y_pred"):
            raise ValueError("Run get_metrics_results() first to populate predictions.")
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        labels = ["Cover", "Stego"]
    
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
    
        # ✅ Add model name to title (if provided)
        title = "Confusion Matrix"
        if model_name:
            title += f" – {model_name}"
        plt.title(title)
    
        if save_path:
            plt.savefig(
                f"{save_path}/confusion_matrix_{model_name or 'model'}.png",
                bbox_inches="tight"
            )
        plt.close()
        return cm
    
    def compare_integrated_gradients(self,models_dict, dataloader, device, num_samples=3, save_path=None):
    
        # Collect random samples
        images_list, labels_list = [], []
        for images, labels in dataloader:
            images_list.append(images)
            labels_list.append(labels)
        images_all = torch.cat(images_list)
        labels_all = torch.cat(labels_list)

        # Ensure save directory exists
        if save_path:
            os.makedirs(save_path, exist_ok=True)

        # Pick random indices
        idxs = np.random.choice(len(images_all), num_samples, replace=False)

        for i, idx in enumerate(idxs):
            img = images_all[idx:idx+1].to(device)
            label = labels_all[idx].item()

            fig, axes = plt.subplots(1, len(models_dict) + 1, figsize=(5 * (len(models_dict) + 1), 5))

            # Plot the original image
            ax = axes[0]
            ax.imshow(img.squeeze().cpu().numpy(), cmap='gray')
            ax.set_title(f"Original (Label={label})")
            ax.axis('off')

            # For each model
            for j, (model_name, model) in enumerate(models_dict.items(), start=1):
                model.eval()
                model.to(device)

                ig = IntegratedGradients(model)
                img.requires_grad_()

                # Compute Integrated Gradients
                attributions, delta = ig.attribute(
                    img, target=None, return_convergence_delta=True
                )

                # Get prediction
                with torch.no_grad():
                    output = model(img)
                    pred = torch.argmax(output, dim=1).item()

                # Normalize attributions
                attr = attributions.squeeze().cpu().detach().numpy()
                attr_norm = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)

                # Blend grayscale with colorful heatmap
                img_np = img.squeeze().cpu().numpy()
                overlay = np.stack([img_np, img_np, img_np], axis=-1)
                heatmap = plt.cm.jet(attr_norm)[..., :3]
                blended = 0.4 * overlay + 0.6 * heatmap

                # Plot attribution map
                ax = axes[j]
                ax.imshow(blended)
                ax.set_title(f"{model_name}\nPred: {pred}")
                ax.axis('off')

            plt.tight_layout()

            if save_path:
                file_name = f"integrated_gradients_sample_{i+1}.png"
                plt.savefig(os.path.join(save_path, file_name), bbox_inches='tight')
                plt.close()
            else:
                plt.show()