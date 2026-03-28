import contextlib
import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.model_selection import ParameterGrid
from sklearn.utils.validation import check_is_fitted
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class DISWOTNAS(BaseEstimator, MetaEstimatorMixin):
    """
    Meta Estimator for Neural Architecture Search using DISWOT similarity metrics.
    
    This estimator searches for the best student architecture by comparing
    feature similarities with a teacher model using DISWOT metrics.
    
    Parameters
    ----------
    estimator : sklearn estimator
        Base estimator to be used for architecture search
        
    param_grid : dict
        Dictionary with parameters names as keys and lists of parameter
        settings to try as values. This defines the architecture search space.
        
    teacher_model : torch.nn.Module
        Pre-trained teacher model for similarity comparison
        
    metric : str, default='relation'
        Similarity metric to use: 'relation' or 'semantic'
        
    batch_size : int, default=32
        Batch size for similarity computation
        
    device : str, default='cuda'
        Device to run computations on ('cuda' or 'cpu')
        
    n_jobs : int, default=1
        Number of parallel jobs (currently only 1 is supported)
        
    verbose : int, default=0
        Verbosity level
    """
    
    def __init__(self, estimator, param_grid, teacher_model, 
                 metric='relation', batch_size=32, device='cuda',
                 n_jobs=1, verbose=0):
        self.estimator = estimator
        self.param_grid = param_grid
        self.teacher_model = teacher_model
        self.metric = metric
        self.batch_size = batch_size
        self.device = device
        self.n_jobs = n_jobs
        self.verbose = verbose
        
    def fit(self, X, y, **fit_params):
        """
        Run architecture search to find the best model configuration.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
            
        y : array-like of shape (n_samples,)
            Target values
            
        **fit_params : dict
            Additional parameters passed to the fit method of the estimator
            
        Returns
        -------
        self : object
            Returns self with best_estimator_ and best_params_ attributes
        """
        # Convert to PyTorch DataLoader if needed
        if isinstance(X, torch.Tensor) or isinstance(X, np.ndarray):
            dataset = torch.utils.data.TensorDataset(
                torch.tensor(X).float(), 
                torch.tensor(y).long()
            )
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=True
            )
        else:
            dataloader = X  # Assume X is already a DataLoader
        
        # Move teacher model to device and set to eval mode
        self.teacher_model = self.teacher_model.to(self.device)
        self.teacher_model.eval()
        
        # Generate all parameter combinations
        param_combinations = list(ParameterGrid(self.param_grid))
        
        if self.verbose > 0:
            print(f"Searching over {len(param_combinations)} configurations")
        
        # Store results
        self.cv_results_ = {
            'params': [],
            'mean_test_score': [],
            'rank_test_score': []
        }
        
        best_score = -float('inf')
        best_params = None
        best_estimator = None
        
        # Search over parameter grid
        for idx, params in enumerate(param_combinations):
            if self.verbose > 0:
                print(f"\nEvaluating configuration {idx + 1}/{len(param_combinations)}")
                print(f"Parameters: {params}")
            
            # Clone estimator with new parameters
            estimator = clone(self.estimator)
            estimator.set_params(**params)
            
            # Convert sklearn estimator to PyTorch model if needed
            student_model = self._estimator_to_model(estimator)
            student_model = student_model.to(self.device)
            
            # Compute similarity score
            score = self._compute_similarity_score(student_model, dataloader)
            
            # Store results — always as plain Python float
            self.cv_results_['params'].append(params)
            self.cv_results_['mean_test_score'].append(score)
            
            if self.verbose > 0:
                print(f"Similarity score: {score:.4f}")
            
            # Update best
            if score > best_score:
                best_score = score
                best_params = params
                best_estimator = estimator
        
        # Store best results as plain Python float
        self.best_score_ = float(best_score)
        self.best_params_ = best_params
        self.best_estimator_ = best_estimator
        
        # Fit best estimator on full data
        if self.verbose > 0:
            print(f"\nBest configuration found with score: {best_score:.4f}")
            print(f"Best parameters: {best_params}")
            print("\nFitting best estimator on full dataset...")
        
        self.best_estimator_.fit(X, y, **fit_params)
        
        # Add ranking
        scores = np.array(self.cv_results_['mean_test_score'], dtype=float)
        self.cv_results_['rank_test_score'] = np.argsort(np.argsort(-scores)) + 1
        
        return self
    
    def predict(self, X):
        """Predict using the best found estimator."""
        check_is_fitted(self, 'best_estimator_')
        return self.best_estimator_.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities using the best found estimator."""
        check_is_fitted(self, 'best_estimator_')
        return self.best_estimator_.predict_proba(X)
    
    def score(self, X, y):
        """Score using the best found estimator."""
        check_is_fitted(self, 'best_estimator_')
        return self.best_estimator_.score(X, y)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {
            'estimator': self.estimator,
            'param_grid': self.param_grid,
            'teacher_model': self.teacher_model,
            'metric': self.metric,
            'batch_size': self.batch_size,
            'device': self.device,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose
        }
        
        if deep and hasattr(self, 'best_estimator_'):
            deep_items = self.best_estimator_.get_params(deep=deep)
            params.update(deep_items)
            
        return params
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        if 'estimator' in params:
            self.estimator = params.pop('estimator')
        if 'param_grid' in params:
            self.param_grid = params.pop('param_grid')
        if 'teacher_model' in params:
            self.teacher_model = params.pop('teacher_model')
            
        for key, value in params.items():
            setattr(self, key, value)
            
        return self
    
    def _estimator_to_model(self, estimator):
        """
        Convert sklearn estimator to PyTorch model.
        """
        if hasattr(estimator, 'to_pytorch_model'):
            return estimator.to_pytorch_model()
        return estimator
    
    def _compute_similarity_score(self, student_model, dataloader):
        """Compute similarity score between teacher and student.

        FIX: The semantic metric calls .backward() and therefore requires an
        active autograd graph.  We use torch.no_grad() only for the relation
        metric, and contextlib.nullcontext() for semantic so that gradients
        are tracked correctly.
        """
        # semantic metric needs the autograd graph for .backward()
        # relation metric is pure forward-pass and benefits from no_grad
        ctx = (torch.no_grad() if self.metric == 'relation'
               else contextlib.nullcontext())

        # For semantic metric the models must be in train() mode so that
        # parameter gradients are computed; relation metric uses eval().
        if self.metric == 'semantic':
            self.teacher_model.train()
            student_model.train()
        else:
            student_model.eval()

        total_similarity = 0.0
        n_batches = 0

        with ctx:
            for batch_data in dataloader:
                if isinstance(batch_data, (list, tuple)):
                    images = batch_data[0]
                    labels = batch_data[1] if len(batch_data) > 1 else None
                else:
                    images = batch_data
                    labels = None
                
                images = images.to(self.device)
                if labels is not None:
                    labels = labels.to(self.device)
                
                if self.metric == 'relation':
                    similarity = relation_similarity_metric(
                        self.teacher_model, student_model, (images, labels)
                    )
                elif self.metric == 'semantic':
                    similarity = semantic_similarity_metric(
                        self.teacher_model, student_model, (images, labels)
                    )
                else:
                    raise ValueError(f"Unknown metric: {self.metric}")
                
                # FIX: always accumulate as plain Python float to avoid
                # keeping the entire computation graph alive across batches
                total_similarity += float(similarity)
                n_batches += 1
        
        # FIX: return plain Python float so best_score_ and cv_results_ are
        # always float, not torch.Tensor
        return total_similarity / n_batches


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def relation_similarity_metric(teacher, student, batch_data):
    """Compute relation similarity between teacher and student."""
    image, label = batch_data
    
    if hasattr(teacher, 'forward_features'):
        t_feats = teacher.forward_features(image)
        s_feats = student.forward_features(image)
    else:
        t_out = teacher(image)
        s_out = student(image)
        return -1 * torch.nn.functional.mse_loss(t_out, s_out)
    
    t_feat = t_feats[-2] if isinstance(t_feats, (list, tuple)) else t_feats
    s_feat = s_feats[-2] if isinstance(s_feats, (list, tuple)) else s_feats
    
    return -1 * batch_similarity(t_feat, s_feat)


def batch_similarity(f_t, f_s):
    """Compute batch-wise similarity matrix distance."""
    bsz = f_t.shape[0]
    
    f_s = f_s.view(f_s.shape[0], -1)
    f_t = f_t.view(f_t.shape[0], -1)
    
    G_s = torch.mm(f_s, torch.t(f_s))
    G_s = F.normalize(G_s, dim=1)
    G_t = torch.mm(f_t, torch.t(f_t))
    G_t = F.normalize(G_t, dim=1)
    
    G_diff = G_t - G_s
    return (G_diff * G_diff).view(-1, 1).sum() / (bsz * bsz)


def semantic_similarity_metric(teacher, student, batch_data):
    """Compute semantic similarity using Grad-CAM.

    NOTE: must NOT be called inside torch.no_grad() — it uses .backward().
    Both models must be in train() mode so parameter grads are computed.
    """
    criterion = nn.CrossEntropyLoss()
    image, label = batch_data
    
    for param in teacher.parameters():
        param.requires_grad = True
    for param in student.parameters():
        param.requires_grad = True
    
    # Zero out stale gradients from previous batches
    teacher.zero_grad()
    student.zero_grad()

    t_logits = teacher(image)
    s_logits = student(image)
    
    criterion(t_logits, label).backward(retain_graph=True)
    criterion(s_logits, label).backward()
    
    t_grad_cam = None
    s_grad_cam = None
    
    for attr in ['fc', 'classifier', 'head', 'linear']:
        if hasattr(teacher, attr) and hasattr(getattr(teacher, attr), 'weight'):
            if t_grad_cam is None:
                t_grad_cam = getattr(teacher, attr).weight.grad
        if hasattr(student, attr) and hasattr(getattr(student, attr), 'weight'):
            if s_grad_cam is None:
                s_grad_cam = getattr(student, attr).weight.grad
    
    if t_grad_cam is None or s_grad_cam is None:
        return relation_similarity_metric(teacher, student, batch_data)
    
    return -1 * channel_similarity(t_grad_cam, s_grad_cam)


def channel_similarity(f_t, f_s):
    """Compute channel-wise similarity matrix distance."""
    bsz, ch = f_s.shape[0], f_s.shape[1]
    
    f_s = f_s.view(bsz, ch, -1)
    f_t = f_t.view(bsz, ch, -1)
    
    emd_s = torch.bmm(f_s, f_s.permute(0, 2, 1))
    emd_s = F.normalize(emd_s, dim=2)
    emd_t = torch.bmm(f_t, f_t.permute(0, 2, 1))
    emd_t = F.normalize(emd_t, dim=2)
    
    G_diff = emd_s - emd_t
    return (G_diff * G_diff).view(bsz, -1).sum() / (ch * bsz)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.svm import SVC
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    class DummyTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(20, 2)
        
        def forward(self, x):
            return self.fc(x)
        
        def forward_features(self, x):
            return [x, self.fc(x)]
    
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    teacher = DummyTeacher()
    
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
    
    nas = DISWOTNAS(
        estimator=SVC(),
        param_grid=param_grid,
        teacher_model=teacher,
        metric='relation',
        batch_size=32,
        device='cpu',
        verbose=1
    )
    
    nas.fit(X_train, y_train)
    
    print(f"\nBest parameters: {nas.best_params_}")
    print(f"Best score: {nas.best_score_}")
    print(f"Test score: {nas.score(X_test, y_test)}")
    
    print("\nAll results:")
    for params, score, rank in zip(
        nas.cv_results_['params'],
        nas.cv_results_['mean_test_score'],
        nas.cv_results_['rank_test_score']
    ):
        print(f"  Rank {rank}: {params} -> {score:.4f}")
