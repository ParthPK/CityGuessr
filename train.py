import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, CosineSimilarity
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import shutil
import wandb

from dataloader import YouTubeVideoDataset
from evaluate import evaluate_hierarchical

def train_model(model, txt_file, val_txt_file, num_epochs, lr, batch_size, transform=None, checkpoint=None, opt=None):
    # Create dataset and dataloader
    print('Train set => ', end='')
    dataset = YouTubeVideoDataset(txt_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print('Val set => ', end='')
    val_dataset = YouTubeVideoDataset(val_txt_file, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()
    text_criterion = CosineSimilarity()

    best_acc1 = 0.0
    start_epoch = 0
    
    if opt.resume:
        ckpt = torch.load(checkpoint)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        best_acc1 = ckpt['best_acc1']
    
    if opt.mode == 'eval':
        val_loss, acc1, acc5 = evaluate_hierarchical(model, val_dataloader, criterion, opt)
        
        return
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for i, (videos, labels) in enumerate(dataloader):
                videos = videos.cuda()
                class_labels = [label.cuda() for label in labels[:-2]]
                texts = torch.transpose(torch.stack(labels[-2]), 0, 1).cuda()
                scenes = torch.transpose(torch.stack(labels[-1]), 0, 1).cuda()

                loss = 0

                # Forward pass
                if opt.model == 'GeoDecoder':
                    outputs, scene_preds, _ = model(videos)
                elif opt.model == 'VanillaTextScenes':
                    outputs, scene_preds, text_preds = model(videos)
                elif opt.model == 'VanillaText':
                    outputs, text_preds = model(videos)
                elif opt.model == 'ISNs':
                    outputs = model(videos,scenes)
                elif opt.model in ['Vanilla', 'Experts', 'SkipExperts']:
                    outputs = model(videos)
                    
                # Hierarchical loss
                loss += sum(criterion(output, label) for output, label in zip(outputs, class_labels))

                # Scene loss
                if opt.model in ['GeoDecoder', 'MrWorldwide', 'VanillaScenes', 'VanillaTextScenes']:
                    loss += criterion(scene_preds, scenes)
                    
                if opt.model in ['VanillaTextScenes','VanillaText']:
                    loss += -1*torch.sum(text_criterion(text_preds, texts))    

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i % opt.log_every == 0:
                    wandb.log({"Training Loss": loss.item()})

                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                pbar.update(1)

        epoch_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")
        
        # val_loss, acc1, acc5 = evaluate_hierarchical(model, val_dataloader, criterion, opt)

        if epoch % opt.eval_every == 0:
            val_loss, acc1, acc5 = evaluate_hierarchical(model, val_dataloader, criterion, opt)
        
        city_acc1 = acc1[-1]
        is_best = city_acc1 > best_acc1
        best_acc1 = max(city_acc1, best_acc1)
        save_checkpoint({
            'epoch' : epoch+1,
            'state_dict' : model.state_dict(),
            'best_acc1' : best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename='checkpoint.pth.tar')
        
        wandb.log({"City Accuracy": city_acc1})
    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
