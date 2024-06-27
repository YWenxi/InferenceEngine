from pLogicNet.kge.model import KGEModel

import torch.nn as nn
import torch


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
        hidden_dim=500,
        gamma=12.0,
        double_entity_embedding=True,
        double_relation_embedding=True
    ) -> None:
        super(pLogicNet, self).__init__()
        self.model = KGEModel(model_name, nentity, nrelation, hidden_dim, 
                                gamma, double_entity_embedding, 
                                double_relation_embedding)
        
        
    def train(self, dataloader, args,
              warmup_steps=None, optimizer="adam", lr=0.001,
              max_steps=int(1E4), cuda=False):
        
        if not isinstance(self.model, KGEModel):
            raise ValueError(f"self.model is not a Valid KGE Model: {self.model}")
        
        # cuda
        if cuda:
            self.cuda()
        
        # init checkpoint
        init_step = 0
        
        # record
        
        # init training logs
        training_logs = []
        
        # optimizer
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr
        )
        
        for i in range(init_step, max_steps):
            print("training step: ", i)
            logs = self.model.train_step(self.model, optimizer, dataloader, args)
            training_logs.append(logs)
            
        return training_logs
    
    
if __name__ == "__main__":
    pass
