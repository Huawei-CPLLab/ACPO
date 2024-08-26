import torch
import torch.nn

def MSELoss_norm(output, target):
    loss_per_sample = torch.nn.MSELoss(reduction='none')(output, target)
    loss_per_sample_normalized = loss_per_sample / target
    return torch.mean(loss_per_sample_normalized)

def L1Loss_norm(output, target):
    loss_per_sample = torch.nn.L1Loss(reduction='none')(output, target)
    loss_per_sample_normalized = loss_per_sample / target
    return torch.mean(loss_per_sample_normalized)

def LogRegLoss(output, target):
    return torch.mean((output - target)*torch.log(target/output) + (target - output)*torch.log(output/target))

def BCELoss_speedup(output, target):
    # output = torch.clamp(output / 2, min=0.0, max=1.0) 
    output = torch.gt(output, 1).float()
    # target = torch.clamp(target / 2, min=0.0, max=1.0) 
    target = torch.gt(target, 1).float()
    return torch.nn.BCELoss()(output, target)

def L1Loss_norm_plus_CrossEntropy(output, target):
    return L1Loss_norm(output, target) + BCELoss_speedup(output, target)

# loss functions
loss_dict = {
    'mse': torch.nn.MSELoss(), 
    'mae': torch.nn.L1Loss(),
    'mse_norm': MSELoss_norm,
    'mae_norm': L1Loss_norm,
    'log_reg': LogRegLoss,
    'cross_entropy' : torch.nn.CrossEntropyLoss(), 
}