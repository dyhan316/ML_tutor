import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

"""
뭐가 여러번 돌아가는지 아는 법 : 사실 "__main__"빼고는 다 여러번 돌아가는 것 같기는 한데, "rank" (or equivalent, gpu_id)가 들어간 놈이면 모두다 여러번 돌아가는 것이라고 이해해도 될듯!

"""


#===========ADDED=================#
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"     #IP address of the machine (localhost b/c I'll only use this computer only)
    os.environ["MASTER_PORT"] = "12355"         #port to be used from the machine
                                                #called "masetr" because it is the machine that coordinates communication b/w proceses
    #initializes the default distributed process group (nccl 이라는 backend로 여기서 설정하자)(nccl : gpu backends)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
#===================================#



###ASK ASK ASK
#그러면 밑에서 어느것이 여러번 작동하고 어느것이 한번 작동하는 거지?? 잘 이해 안됨...
#########
    
#===========CHANGED=================#
#===================================#
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        
        #==========ADDED=================#
        self.model = DDP(model, device_ids=[gpu_id]) #wrap it with DDP before running the model
                                    #device_ids : list of gpu ids that the model lives on ([0,1] in our case)
        #===================================#
        
        
    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        
        
        ####ASK ASK AKS###
        #===========ADDED===================#
        self.train_data.sampler.set_epoch(epoch) #has to be added because if not, the same ordering will be used in each epoch
                            #이 파트 잘 이해안됨
        #===================================#
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        #===========CHANGED=================#
        #원래 ckp = self.model.state_dict()
        
        ckp = self.model.module.state_dict() 
        """이렇게 바뀐이유 : 원래는 model 자체가 model이여서 self.model.state_dict()하면 됬었는데,
        model에 wrapper 을 씌워주었기에, self.model 을 하면 DDP wrapping된 것이나옴!
        따라서, DDP에서 (기존)model만을 끄집어 내기 위해서 `.module` 로추가적으로 꺼내줘야함
        따라서, `model.module` 인 것!"""
        #===================================#
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            #===========CHANGED=================#
            #원래 : 
            #if epoch % self.save_every ==0:
            #    self._save_checkpoint(epoch) 
            
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
            
            """이렇게 바꾼이유 : 원래는 epoch이 맞기만 하면 save하면 됬었는데, DDP에서는 process가 여러개 돌아가기에, 
            each gpu makes : one process => one trainer class => one save_checkpoint
               
            이것이 여러번 되기에, 불필요하게 torch save되는 것이 좋지 않다! 
            따라서, 이것을 막기 위해서 rank=0 process에서만 torch.save가 되도록 하자!            
            """
            #===================================#


def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        
        #===========CHANGED=================#
        #원래 : shuffle = True,
        shuffle=False, #changed because we're already using sampler to distribvute the samples
        #===================================#
        
        #===========ADDED===================#
        sampler=DistributedSampler(dataset)   #added so that we can divide the batch and send it to each processes #여기다는 gpu몇개인지/rank정보를 안줘도 알아서 하는 듯?
        #===================================#
    )


#===========CHANGED=================#
#원래 : def main(device: int, total_epochs: int, batch_size: int):

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    #changed because we need to provde the rank and other stuff
#===================================#
    #==========ADDED====================#
    ddp_setup(rank, world_size)             #initialize the DDP group (which we defined as ddp_setup)
    #===================================#
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size) #modified dataloader due to different sampler
    trainer = Trainer(model, train_data, optimizer, rank, save_every) #이것도 'device'에서 'rank'로 바뀜 
    trainer.train(total_epochs)
    
    #==========ADDED====================#
    destroy_process_group()                
    #added to destroy the process group once it's done
    #===================================#

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    

    #==========ADDED====================#
    world_size = torch.cuda.device_count() #added so that we an take the world_size below
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)      #i.e. spawn multiprocessing (i.e. run the func "main"), and with put "args" as the input to the main()함수 (이때 spawn시 rank는 안넣어도 mp.spawn이 알아서 만들어줌), with total of nproc 갯수 의 processes
    #===================================#