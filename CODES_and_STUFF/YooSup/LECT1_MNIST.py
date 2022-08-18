## package 추가하기
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter #have to download this

from torchvision import transforms, datasets

print("===tensorboard not downloaded... find out if nightly can download it")

lr = 1e-3
batch_size = 64
num_epoch = 10

ckpt_dir = './checkpoint' #checkpoint라는데 무슨 용도로 쓰일지는 아직 모르겠다
log_dir = './log'         #for loggoing i think ?

device = torch.device('mps' if torch.backends.mps.is_available() else "cpu")

print("using the device : ", device)

##make network
class Net(nn.Module):
    def __int__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size= 5, stride= 1, padding=0, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=10 ,out_channels=20, kernel_size=5, stride=1, padding=0, bias=True)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(in_features=320, out_features=50, bias=True)
        self.relu_fc1 = nn.ReLU()
        self.drop1_fc1 = nn.Dropout2d(p = 0.5)

        self.fc2 = nn.Linear(in_features=50, out_features=10, bias=True)

    def forward(self, x):
        x = self.relu1(self.pool1(self.conv1(x)))

        x = self.relu2(self.pool2(self.drop2(self.conv2(x))))

        x = x.view(-1, 320) #flattening

        x = self.fc2(self.drop1_fc1(self.relu_fc1(self.fc1(x))))

        return x

###below : learn
#.format이 아닌, 새로운 방식으로 string에 넣는 방법 (with the use of %s, % and so on)
#model.state_dict(), optimizer.state_dict()를 해서 state dictionary를 가져오기 가능!! (optimizer만 가져오면 안됨)


#saving the network (network saving해주는 함수 구현)
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

        #아! dictionary 형태로 저장하는 것! (net이라는 keywoard에는 state_dict가 있고, and so on)
    torch.save({'net':net.state_dict(), 'optim' : optim.state_dict()},
               './%s/model_epoch%d.pth' % (ckpt_dir, epoch))

#MNIST dataloading하기
def load(ckpt_dir, net, optim):
    ckpt_list = os.listdir(ckpt_dir) #i.e. checkpoint directory 안에 있는 것들을 list화해서 이름 저장
    ckpt_list.sort()                #epoch별로 sorting하기 (b/c 이름뒤에 숫작(epoch) 이 붙어있으니)

    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_list[-1])) #즉, 맨 마지막 epoch 의 model state_dict 들을 저장했던
                                    #dictionary를 불러오기

    net.load_state_dict(dict_model['net']) #net, optim의 state_dict을 저장한 dict에서 net의 것을 가져와서 net라는 model 로 loading하기
    optim.load_state_dict(dict_model['optim']) #same for the optimizer

    return net, optim

##MNIST 데이터 불러오기
#먼저, dataloading시 사용할 transformation 정의해주기
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
    #transforms.compose : transforms여러개를 묶어주는 것 (PIL => tensor => normalize to values) (이것을 nn.Sequential처럼 transforms 로 묶을때
    #transforms.Compose를 쓴다
#downloading the MNISt dataset with the predefined transformation
dataset = datasets.MNIST(download=True, root = './', train = True, transform=transform) #위에서 정의한 transform으로 dataset을 정의해주기
loader = DataLoader(dataset, batch_size = batch_size, shuffle= True, num_workers= 0) #batch_size : 64


num_data = len(loader.dataset) #number of samples
num_batch = np.ceil(num_data / batch_size) #i.e. epoch한번에 iteration (batch) 몇번 돌리는지?

##loss function, optimizer definition
net = Net().to(device)
print("Net : ", net)
params = net.parameters()

#여러가지 "함수"(method)들 (loss, prediction등등) 정의해놓기
fn_loss = nn.CrossEntropyLoss().to(device) #model자체에서 softmax 안해서 여기서 한다 #이것도 device로 보내야함
fn_pred = lambda output : torch.softmax(output, dim=1) #i.e. "output"을 input으로 받는 함수를 fn_pred로 정의
                                                #softmax 를 통해서 probabilty (logit)으로 바꿔준다
fn_acc = lambda pred, label : ((pred.max(dim=1)[1] == label).type(torch.float)).mean() #즉 맞으면 1, dkslaus 0, 이것을 one batch 에 대해서 평균을 구하기



print("params : ", params)
optim = torch.optim.Adam(params, lr = lr) #net.parameters() (net의 parameters)"를" Adam optimizer에서 optimzer한다

#writer = SummaryWriter(log_dir = log_dir) #tensorboard에 있는 것 중 하나로, `writer.add_image`이런것들을 나중에 해줄 수 있다

##Start Training with FOR loop
for epoch in range(num_epoch):    #원래는 epoch = 1 에서 시작하도록 되어있었는데, 그렇게 하지 말자. 복잡함 0에서 시작해도 OK
    net.train()    #model을 train mode로 바꿔줘야한다!

    #batch들의 loss/acc값들을 저장해놓기 (epoch마다 reset)
    loss_arr = []
    acc_arr = []

##one epoch : iteration over  batches
    for batch, (image, label) in enumerate(loader):
        #send to GPU first
        image = image.to(device)
        label = label.to(device)

        #prediction, loss 구하기
        output = net(image)
        pred = fn_pred(output)

        loss = fn_loss(output, label) #여기다가 pred대신 loss가 들어가면 안됨!! (이미 nn.CrossEntropyLoss에서 logit을 만들어주기 땜누에)
        acc = fn_acc(pred, label)

        #optimzier zero setting하고 정의하기
        optim.zero_grad() #zero grad로 리셋해야함! (batch 마다 grad를 다 더해서 하기 때문에)
        loss.backward() #loss내에 내장된 computational graph를 돌리기

        #각 batch마다의 loss, acc값들을 array에 저장해두기
        loss_arr = loss_arr.append(loss.item()) #item() : torch.tensor만 뽑아내기 위해서#이파트 원래꺼랑 조금다름..
        acc_arr = acc_arr.append(acc)

        print('TRAIN: EPOCH %04d/%04d | BATCH %04d/%04d | LOSS: %.4f | ACC %.4f' %
              (epoch, num_epoch, batch, num_batch, np.mean(loss_arr), np.mean(acc_arr)))

    #batch들의 loss, acc가 된 것을 평균시켜서 해당 epoch의 loss, acc값들로 writer로 저장하기
    #writer.add_scalar('loss', np.mean(loss_arr), epoch) ##즉, one epoch에 대한 iteration들의 평균값 구하기 (batch들의 평균값)
    #writer.add_scalar('acc', np.mean(acc_arr),epoch)

    save(ckpt_dir = ckpt_dir, net = net, optim = optim, epoch = epoch) #위에서 정의했던 save함수를 써서 state dict들을 정의하기

#writer.close()


##위에 나오는 것 이해하기
#a = torch.rand((2,4)) #2 : batch 갯수 ,4 : dimension
#torch.softmax(a, dim = 1) #결과 : probabilty vector
#위의 결과와 ,label :



####lambda공부하기
#a = lambda x : x**2
#a(3) 하면 9로 나옴 (즉, 함수를 정의해주는 것)

#define the loss thing


###things to see
ho = "hohoho"
print("%s hihi" % ho )
#print("%d hihi" % ho ) 에러가 뜬다 (becasue %d means a number should be input)


print(batch_size)
        
