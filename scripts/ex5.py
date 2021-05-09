## excludes data preprocessing, augmentation 
## excludes model evaluation

class CXRImgClf(nn.Module):
    def __init__(self):
        super(CXRImgClf, self).__init__()
        self.network= nn.Sequential(
            nn.Conv2d(3,8,3), 
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(start_dim=1), #.view(-1,)
            nn.Linear(8*121*121,8*11*11),
            nn.Linear(8*11*11, 8*2*2),
            nn.Linear(8*2*2, 4)
        )
        
    def forward(self, x):
        return self.network(x)

def train_model(model, train_dataloader, n_epoch=n_epochs, optimizer=optimizer, criterion=criterion):
    import torch.optim as optim
    """
    :param model: A CNN model
    :param train_dataloader: the DataLoader of the training data
    :param n_epoch: number of epochs to train
    :return:
        model: trained model
    TODO:
        Within the loop, do the normal training procedures:
            pass the input through the model
            pass the output through loss_func to compute the loss (name the variable as *loss*)
            zero out currently accumulated gradient, use loss.basckward to backprop the gradients, then call optimizer.step
    """
    model.train() # prep model for training
    
    
    for epoch in range(n_epoch):
        curr_epoch_loss = []
        for data, target in train_dataloader:
            # your code here
            #raise NotImplementedError
            # pass the input through the model
            data = data.to(device)
            target = target.to(device)
            out_train = model(data)
            out_soft = torch.softmax(out_train, dim=1)
            
            # pass the output through loss_func
            loss = criterion(out_train, target)
            # zero out currently accumulated gradient, 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            curr_epoch_loss.append(loss.cpu().data.numpy())
        print(f"Epoch {epoch}: curr_epoch_loss={np.mean(curr_epoch_loss)}")
    return model