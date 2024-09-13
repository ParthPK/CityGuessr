import time
import json
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Resize, ToTensor, transforms
import numpy as np
from tqdm import tqdm
from dataloader import YouTubeVideoDataset
from pathlib import Path
from networks.ISNs import ISNsVideo
from config import get_opt

def evaluate_hierarchical(model, dataloader, criterion, opt=None):
    model.eval()
    total_continent = 0
    total_country = 0
    total_state = 0
    total_city = 0
    top1_correct_continent = 0
    top1_correct_country = 0
    top1_correct_state = 0
    top1_correct_city = 0
    top5_correct_continent = 0
    top5_correct_country = 0
    top5_correct_state = 0
    top5_correct_city = 0

    with open('misc/hierarchy_mappings.json', 'rb') as f:
        mapping = json.load(f)
        city_to_state = mapping[0]
        state_to_country = mapping[1]
        country_to_continent = mapping[2]

    ts = time.time()

    '''
    preds_city = []
    preds_state = []
    preds_country = []
    preds_continent = []
    gt_city = []
    gt_state = []
    gt_country = []
    gt_continent = []

    map_logits_city = []

    logits_state = []
    logits_country = []
    logits_continent = []
    '''
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Evaluation") as pbar:
            for videos, labels in dataloader:
                videos = videos.cuda()
                class_labels = [label.cuda() for label in labels[:-2]]
                texts = torch.transpose(torch.stack(labels[-2]), 0, 1).cuda()
                scenes = torch.transpose(torch.stack(labels[-2]), 0, 1).cuda()
            
                # Forward pass
                if opt.model == 'GeoDecoder':
                    outputs, scene_preds, _ = model(videos)
                elif opt.model == 'VanillaScenes':
                    outputs, scene_preds = model(videos)
                elif opt.model == 'VanillaTextScenes':
                    outputs, scene_preds, text_preds = model(videos)
                elif opt.model == 'VanillaText':
                    outputs, text_preds = model(videos)
                elif opt.model == 'ISNs':
                    outputs = model(videos,scenes)
                elif opt.model in ['Vanilla', 'Experts', 'SkipExperts']:
                    outputs = model(videos)

                city_logits, state_logits, country_logits, continent_logits = outputs

                # Apply softmaz
                city_logits = nn.functional.softmax(city_logits, dim=1)
                state_logits = nn.functional.softmax(state_logits, dim=1)
                country_logits = nn.functional.softmax(country_logits, dim=1)
                continent_logits = nn.functional.softmax(continent_logits, dim=1)

                '''
                country_to_continent_weights = torch.ones(continent_logits.shape[0], country_logits.shape[1]).cuda()
                for i in range(country_logits.shape[1]):
                    country_to_continent_weights[:,i] = continent_logits[:, country_to_continent[str(i)]]
                country_logits *= country_to_continent_weights
                #print(country_to_continent_weights.shape)
                
                state_to_country_weights = torch.ones(country_logits.shape[0], state_logits.shape[1]).cuda()
                for i in range(state_logits.shape[1]):
                    state_to_country_weights[:,i] = country_logits[:, state_to_country[str(i)]]
                state_logits *= state_to_country_weights
                #print(state_to_country_weights.shape)

                city_to_state_weights = torch.ones(state_logits.shape[0], city_logits.shape[1]).cuda()
                for i in range(city_logits.shape[1]):
                    city_to_state_weights[:,i] = state_logits[:, city_to_state[str(i)]]
                city_logits *= city_to_state_weights
                #print(city_to_state_weights.shape)

                '''
                '''
                map_logits_city.append(city_logits)

                logits_state.append(state_logits)
                logits_country.append(country_logits)
                logits_continent.append(continent_logits)
                '''
                loss = sum(criterion(output, label) for output, label in zip(outputs, class_labels))

                # Calculate top-1 accuracy
                city_predictions = city_logits.argmax(dim=1)
                
                if opt.hierarchical_mode == 'independent':
                    continent_predictions = continent_logits.argmax(dim=1)
                    country_predictions = country_logits.argmax(dim=1)
                    state_predictions = state_logits.argmax(dim=1)
                
                elif opt.hierarchical_mode == 'codependent':
                    state_predictions = torch.zeros_like(city_predictions).cuda()
                    country_predictions = torch.zeros_like(city_predictions).cuda()
                    continent_predictions = torch.zeros_like(city_predictions).cuda()

                    for i in range(city_predictions.shape[0]):
                        state_predictions[i] = city_to_state[str(city_predictions[i].item())]
                        country_predictions[i] = state_to_country[str(state_predictions[i].item())]
                        continent_predictions[i] = country_to_continent[str(country_predictions[i].item())]

                total_continent += class_labels[3].size(0)
                total_country += class_labels[2].size(0)
                total_state += class_labels[1].size(0)
                total_city += class_labels[0].size(0)
                '''
                preds_city.append(city_predictions)
                preds_state.append(state_predictions)
                preds_country.append(country_predictions)
                preds_continent.append(continent_predictions)
                gt_city.append(class_labels[0])
                gt_state.append(class_labels[1])
                gt_country.append(class_labels[2])
                gt_continent.append(class_labels[3])
                '''

                # Calculate top-1 correct predictions for each category
                top1_correct_continent += continent_predictions.eq(class_labels[3]).sum().item()
                top1_correct_country += country_predictions.eq(class_labels[2]).sum().item()
                top1_correct_state += state_predictions.eq(class_labels[1]).sum().item()
                top1_correct_city += city_predictions.eq(class_labels[0]).sum().item()

                # Calculate top-5 correct predictions for each category
                _, continent_top5_predictions = continent_logits.topk(5, dim=1, largest=True, sorted=True)
                _, country_top5_predictions = country_logits.topk(5, dim=1, largest=True, sorted=True)
                _, state_top5_predictions = state_logits.topk(5, dim=1, largest=True, sorted=True)
                _, city_top5_predictions = city_logits.topk(5, dim=1, largest=True, sorted=True)

                top5_correct_continent += continent_top5_predictions.eq(class_labels[3].view(-1, 1).expand_as(continent_top5_predictions)).sum().item()
                top5_correct_country += country_top5_predictions.eq(class_labels[2].view(-1, 1).expand_as(country_top5_predictions)).sum().item()
                top5_correct_state += state_top5_predictions.eq(class_labels[1].view(-1, 1).expand_as(state_top5_predictions)).sum().item()
                top5_correct_city += city_top5_predictions.eq(class_labels[0].view(-1, 1).expand_as(city_top5_predictions)).sum().item()

                
                
                pbar.set_postfix({'Loss': loss.item()})
                pbar.update(1)

    '''
    pred = np.array([torch.cat(preds_city).cpu(), torch.cat(preds_state).cpu(), torch.cat(preds_country).cpu(), torch.cat(preds_continent).cpu()])
    gt = np.array([torch.cat(gt_city).cpu(), torch.cat(gt_state).cpu(), torch.cat(gt_country).cpu(), torch.cat(gt_continent).cpu()])

    logit_city = np.array(torch.cat(map_logits_city).cpu())

    logit_state = np.array(torch.cat(logits_state).cpu())
    logit_country = np.array(torch.cat(logits_country).cpu())
    logit_continent = np.array(torch.cat(logits_continent).cpu())
    np.save('preds.npy', pred)
    np.save('gt.npy', gt)

    np.save('map_logits_city.npy', logit_city)

    np.save('logits_state.npy', logit_state)
    np.save('logits_country.npy', logit_country)
    np.save('logits_continent.npy', logit_continent)
    '''
    avg_loss = loss / len(dataloader)
    top1_accuracy_continent = 100 * top1_correct_continent / total_continent
    top1_accuracy_country = 100 * top1_correct_country / total_country
    top1_accuracy_state = 100 * top1_correct_state / total_state
    top1_accuracy_city = 100 * top1_correct_city / total_city

    top5_accuracy_continent = 100 * top5_correct_continent / total_continent
    top5_accuracy_country = 100 * top5_correct_country / total_country
    top5_accuracy_state = 100 * top5_correct_state / total_state
    top5_accuracy_city = 100 * top5_correct_city / total_city

    print('City : Percentage-top1:{}, top5:{}'.format(top1_accuracy_city, top5_accuracy_city))
    print('State/Province : Percentage-top1:{}, top5:{}'.format(top1_accuracy_state, top5_accuracy_state))
    print('Country : Percentage-top1:{}, top5:{}'.format(top1_accuracy_country, top5_accuracy_country))
    print('Continent : Percentage-top1:{}, top5:{}'.format(top1_accuracy_continent, top5_accuracy_continent))
    print('Time : {}'.format(time.time() - ts))

    wandb.log({
        "Top 1 Accuracy (Continent)": top1_accuracy_continent,
        "Top 1 Accuracy (Country)": top1_accuracy_country,
        "Top 1 Accuracy (State)": top1_accuracy_state,
        "Top 1 Accuracy (City)": top1_accuracy_city,
        "Top 5 Accuracy (Continent)": top5_accuracy_continent,
        "Top 5 Accuracy (Country)": top5_accuracy_country,
        "Top 5 Accuracy (State)": top5_accuracy_state,
        "Top 5 Accuracy (City)": top5_accuracy_city,
    })

    return avg_loss, [top1_accuracy_continent, top1_accuracy_country, top1_accuracy_state, top1_accuracy_city], [top5_accuracy_continent, top5_accuracy_country, top5_accuracy_state, top5_accuracy_city]

def main(weights_path, eval_file, opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = ISNsVideo().cuda()
    ckpt = torch.load(weights_path)
    model.load_state_dict(ckpt['state_dict'])
    start_epoch = ckpt['epoch']
    best_acc1 = ckpt['best_acc1']
    model.to(device)

    # Define loss criterion
    criterion = nn.CrossEntropyLoss()

    # Data loading for evaluation
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    eval_transform = transforms.Compose([Resize((224, 224)), ToTensor(), Normalize(mean, std)])
    eval_dataset = YouTubeVideoDataset(eval_file, transform=eval_transform)
    eval_dataloader = DataLoader(eval_dataset, batch_size=4, shuffle=False)

    # Evaluation
    eval_loss, top1_accuracy_continent, top1_accuracy_country, top1_accuracy_state, top1_accuracy_city, \
    top5_accuracy_continent, top5_accuracy_country, top5_accuracy_state, top5_accuracy_city = evaluate_hierarchical(model, eval_dataloader, criterion, opt)

    print(f"Average Evaluation Loss: {eval_loss:.4f}")
    print(f"Top-1 Accuracy (Continent): {top1_accuracy_continent:.2f}%")
    print(f"Top-1 Accuracy (Country): {top1_accuracy_country:.2f}%")
    print(f"Top-1 Accuracy (State): {top1_accuracy_state:.2f}%")
    print(f"Top-1 Accuracy (City): {top1_accuracy_city:.2f}%")
    print(f"Top-5 Accuracy (Continent): {top5_accuracy_continent:.2f}%")
    print(f"Top-5 Accuracy (Country): {top5_accuracy_country:.2f}%")
    print(f"Top-5 Accuracy (State): {top5_accuracy_state:.2f}%")
    print(f"Top-5 Accuracy (City): {top5_accuracy_city:.2f}%")

if __name__ == '__main__':
    opt = get_opt()
    weights_path = "/home/parthpk/Serious_projects/ViGeo-main/checkpoint.pth.tar"
    eval_file = "misc/test.txt"
    main(weights_path, eval_file, opt)
