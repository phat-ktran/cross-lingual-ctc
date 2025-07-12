import torch

def build_optimizer(config, parameters):
    """
    Builds an Adam optimizer based on the provided configuration.
    
    Args:
        config (dict): Configuration dictionary containing optimizer settings.
        parameters (iterable): Model parameters to optimize.
    
    Returns:
        torch.optim.Optimizer: Configured Adam optimizer.
    """
    if config['name'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            parameters,
            lr=config['lr']['learning_rate'],           # Base learning rate: 0.001
            betas=(config['beta1'], config['beta2']),   # Betas: (0.9, 0.999)
            weight_decay=config.get('weight_decay', 0)
        )
        return optimizer
    else:
        raise NotImplementedError(f"Optimizer {config['name']} not supported")