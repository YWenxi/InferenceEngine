from pLogicNet.kge.model import KGEModel
from utils import ensure_dir
from utils import set_logger
from utils import save_yaml
from utils import log_metrics

import torch.nn as nn
import torch
import numpy as np

from math import log10

import logging
import os


def save_model(model: nn.Module, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )


class BaseModel:
    def __init__(self) -> None:
        self.model: nn.Module = None
    
    def train(self):
        raise NotImplementedError
    
    def complete(self):
        raise NotImplementedError
    
    def correct(self):
        raise NotImplementedError
    
    def cuda(self):
        self.model.cuda()
    
    
class pLogicNet(BaseModel):
    def __init__(
        self,
        nentity,
        nrelation,
        model_name="TransE",
        hidden_dim=None,
        gamma=12.0,
        double_entity_embedding=True,
        double_relation_embedding=True
    ) -> None:
        super(pLogicNet, self).__init__()
        if hidden_dim is None:
            # then we would choose the hidden dim dynamically
            assert nentity > 0
            hidden_dim = 8 * 2 ** int(log10(nentity))
                
        self.model = KGEModel(model_name, nentity, nrelation, hidden_dim, 
                                gamma, double_entity_embedding, 
                                double_relation_embedding)
        
        
    def train(self, dataloader, workspace, args,
              warm_up_steps=None, optimizer="adam", lr=0.001,
              max_steps=int(1E4), device='cuda',
              log_step=5):
        
        set_logger(ensure_dir(workspace) / "train.log")
        args.save_path = ensure_dir(workspace / "save")
        
        if not isinstance(self.model, KGEModel):
            raise ValueError(f"self.model is not a Valid KGE Model: {self.model}")
        
        logging.info(f"Model: {self.model.model_name}")
        logging.info(f"#entity: {self.model.nentity}")
        logging.info(f"#relation: {self.model.nrelation}")
        logging.info(f"Workspace: {workspace}")
        
        # cuda
        if device == 'cuda' and torch.cuda.is_available and args.cuda:
            self.cuda()
            logging.info(f"Using CUDA {torch.version.cuda}")
        else:
            device = 'cpu'
            args.cuda = False
            logging.info(f"Using CPU.")
            
        # Model Configuration
        logging.info("-"*10 + f'Model Hidden Dim: {self.model.hidden_dim}')
        logging.info("-"*10 + 'Model Parameter Configuration:')
        for name, param in self.model.named_parameters():
            logging.info('Parameter %s: %s, require_grad = %s' 
                         % (name, str(param.size()), str(param.requires_grad)))
        
        # init checkpoint
        init_step = 0
        
        # record
        save_yaml(workspace / "exp_configs.yaml", args)
        
        # init training logs
        training_logs = []
        
        # optimizer
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr
        )
        
        # training config
        logging.info("-"*10 + 'Training Configuration:')
        logging.info('init_step = %d' % init_step)
        logging.info('learning_rate = %d' % lr)
        logging.info('batch_size = %d' % args.batch_size)
        logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
        logging.info('hidden_dim = %d' % self.model.hidden_dim)
        logging.info('gamma = %f' % self.model.gamma)
        logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
        logging.info(f'optimizer: {optimizer}')
        if args.negative_adversarial_sampling:
            logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
        
        # start training
        logging.info("-"*20 + "TRAINING STARTS" + "-"*20)
        for step in range(init_step, max_steps):
            logs = self.model.train_step(self.model, optimizer, dataloader, args)
            training_logs.append(logs)
            
            # log step
            if step % log_step == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []
                
        logging.info("-"*20 + "TRAINING ENDS" + "-"*20)
        
        # save
        save_variable_list = {
            'step': step, 
            'current_learning_rate': lr,
            'warm_up_steps': warm_up_steps
        }
        save_model(self.model, optimizer, save_variable_list, args)
        logging.info(f"training output saved at {args.save_path}")
            
        return training_logs
    
    
if __name__ == "__main__":
    from test_all import test_model
    test_model()