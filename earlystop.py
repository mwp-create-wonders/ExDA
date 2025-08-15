import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    
    def __init__(self, patience=7, delta=0, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                             Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                             Default: 0
            verbose (bool): If True, prints a message for each validation loss improvement.
                             Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss  # Initialize min loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss  # Update min loss
            self.counter = 0

# 使用示例
if __name__ == "__main__":
    early_stopping = EarlyStopping(patience=5, delta=-0.001, verbose=True)
    # 模拟验证损失的变化
    for epoch in range(10):
        val_loss = np.random.rand()  # 模拟验证损失
        early_stopping(val_loss, None)  # 这里的 model 参数可以是 None 或实际模型
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break