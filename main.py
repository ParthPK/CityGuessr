import torch
from torchvision import transforms
import wandb

from networks.ISNs import ISNsVideo
from networks.GeoDecoder import GeoDecoderVideo
from networks.Vanilla import VanillaVideo
from networks.VanillaText import VanillaVideoText
from networks.VanillaTextScenes import VanillaVideoTextScenes
from networks.Experts import ExpertsVideo
from networks.SkipExperts import SkipExpertsVideo

from train import train_model
from dataloader import transform

from config import get_opt


if __name__ == "__main__":
    opt = get_opt()

    wandb.init(project='Mr-Worldwide',
               settings=wandb.Settings(start_method='fork'),
               config=vars(opt))
               
    wandb.run.name = opt.description

    if opt.model == 'ISNs':
        model = ISNsVideo().to(opt.device)
    elif opt.model == 'GeoDecoder':
        model = GeoDecoderVideo().to(opt.device)
    elif opt.model == 'Vanilla':
        model = VanillaVideo().to(opt.device)
    elif opt.model == 'VanillaText':
        model = VanillaVideoText().to(opt.device)
    elif opt.model == 'VanillaTextScenes':
        model = VanillaVideoTextScenes().to(opt.device)
    elif opt.model == 'Experts':
        model = ExpertsVideo().to(opt.device)
    elif opt.model == 'SkipExperts':
        model = SkipExpertsVideo().to(opt.device)

    # Training
    train_model(
        model=model,
        txt_file=opt.txt_file,
        val_txt_file=opt.val_txt_file,
        num_epochs=opt.num_epochs,
        lr=opt.learning_rate,
        batch_size=opt.batch_size,
        transform=transform,
        checkpoint=opt.checkpoint,
        opt=opt,
    )
