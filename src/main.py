import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
import os
from rgbdData import RGBD

class para(object):
    pass

def run(net=None, args=None):
    if args is None:
        args = para
        args.stage='train'
        args.root='../rgbd/'
        args.lr = 0.01
        args.batch_size=128
        args.weight_decay=5e-4
        args.max_epoch=20
        args.exp='./exps/'
        args.resume_path=''

    if not os.path.exists(args.exp):
        os.makedirs(args.exp)

    #mean = [0.485, 0.456, 0.406]
    #std = [0.229, 0.224, 0.225]
    #normalize = torchvision.transforms.Normalize(mean=mean,std=std)
    t = torchvision.transforms.Compose([
                    transforms.Resize((100,100)),
                    torchvision.transforms.ToTensor(),
                    #normalize
            ])

    if args.stage == 'train':
        dataset = RGBD(stage='train', transforms=t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=16, shuffle=True)

        val_dataset = RGBD(stage='val', transforms=t)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=16, shuffle=True)
    else:
        val_dataset = RGBD(stage='val', transforms=t)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=16, shuffle=True)

    if net is None:
        network = rgbdNet()
    else:
        network = net()
    device = torch.device('cuda: 0')
    network.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, network.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)
    if args.stage == 'train':
        if args.resume_path != '' and args.resume_path != 'none':
            state_dict = torch.load(args.resume_path)
            network.load_state_dict(state_dict)
        for i_epoch in range(args.max_epoch):
            epoch_loss = train(i_epoch, network, criterion, optimizer, dataloader, device)
            save_name = 'save_{}.pth'.format(int(i_epoch/5)*5)
            torch.save(network.state_dict(), os.path.join(args.exp, save_name))
            scheduler.step()
            with torch.no_grad():
                eval(network, val_dataloader, device)
    else:
        state_dict = torch.load(args.resume_path)
        network.load_state_dict(state_dict)
        with torch.no_grad():
            network.eval()
            eval(network, dataloader, device)


def train(i_epoch, network, criterion, optimizer, dataloader, device):
    network.train()
    losses = 0.
    pbar = dataloader
    for i_batch, data in enumerate(pbar):
        img = data[0].to(device)
        dep = data[1].float().to(device)/10000.
        label = data[2].long().to(device)
        output = network(img, dep).to(device)

        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss.item()
        lr = optimizer.param_groups[0]['lr']
        
    print("Epoch:{}".format(i_epoch)+'\t loss:{:.4f}, lr:{}'.format(losses/(i_batch+1), lr))
    return losses/(i_batch+1)

def eval(network, dataloader, device):
    network.eval()
    total_correct = 0.
    total_samples = 0.
    pbar = dataloader
    for data in pbar:
        img = data[0].to(device)
        dep = data[1].float().to(device)/10000.
        label  = data[2].long().to(device)

        output = network(img, dep).to(device)

        judge = output.argmax(dim=1)
        correct = (judge == label).sum().item()
        total = output.size()[0]
        total_correct += correct
        total_samples += total
    print('Accuracy:{:.4f}\n'.format(total_correct/total_samples))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--root', default='../rgbd/', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--max_epoch', default=100, type=int)
    parser.add_argument('--exp', default='./exps/', type=str)
    parser.add_argument('--resume_path', default='', type=str)
    global args
    args = parser.parse_args()
    run(args=args)






