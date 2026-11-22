"""SmallCNN reused verbatim from the prior encrypted traffic project."""
from torch import nn

class SmallCNN(nn.Module):
    def __init__(self,classes):
        super().__init__()
        self.body=nn.Sequential(nn.Conv1d(4,32,5,padding=2),nn.ReLU(),nn.Conv1d(32,64,3,padding=1),nn.ReLU(),nn.Conv1d(64,64,3,padding=1),nn.ReLU())
        self.head=nn.Linear(64,classes)
    def forward(self,x):
        mask=x[:,3:4,:]
        return self.head((self.body(x)*mask).sum(2)/mask.sum(2).clamp_min(1))

