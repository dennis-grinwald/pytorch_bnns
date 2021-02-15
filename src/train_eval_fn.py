import numpy as np
from tqdm import tqdm
import torch

def train(model, trainloader, testloader, criterion, optimizer, 
                scheduler, epochs=1,p=0.0, model_save_path='./trash/', device='cpu'):
    
    test_acc = []
    train_acc = []

    best_acc = 0.0
    total_correct = 0.0
    model.train()
    for epoch in range(epochs):
        for images, labels in tqdm(trainloader, position=0, leave=True):
            logits = model(images.to(device))
            y_train_pred =  np.argmax(logits.cpu().detach().numpy(), axis=1)
            total_correct += np.sum(y_train_pred == labels.numpy()) 
            loss = criterion(logits, labels.to(device))
            model.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'\nTraining accuracy after epoch {epoch+1}: {(total_correct/len(trainloader.dataset))*100}%')
        train_acc.append(total_correct/len(trainloader.dataset))
        total_correct = 0.0
        y_pred, targets = eval(model, testloader, device)
        tmp_acc = np.mean(np.argmax(y_pred.cpu().numpy(), axis=1) == targets.numpy())
        test_acc.append(tmp_acc)
        print(f'\nTest accuracy after epoch {epoch+1}: {100 * tmp_acc:.2f}%')
    
    torch.save(model, model_save_path)

    return torch.load(model_save_path), train_acc, test_acc
            
def eval(model, data, device):
    model.eval()
    logits = torch.Tensor().to(device)
    targets = torch.LongTensor()

    with torch.no_grad():
        for images, labels in tqdm(data, position=0, leave=True):
            logits = torch.cat([logits, model(images.to(device))])
            targets = torch.cat([targets, labels])
    return logits, targets

def accuracy(predictions, labels):
    print(f"\nAccuracy: {100 * np.mean(np.argmax(predictions.cpu().numpy(), axis=1) == labels.numpy()):.2f}%")