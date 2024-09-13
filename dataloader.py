import torch
import cv2
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class YouTubeVideoDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.txt_file = txt_file
        self.transform = transform
        self.file_paths, self.city_labels, self.state_labels, self.country_labels, self.continent_labels, self.text_labels, self.scene_labels = self._parse_txt_file()

    def _parse_txt_file(self):
        file_paths = []
        city_labels = []
        state_labels = []
        country_labels = []
        continent_labels = []
        scene_labels = []
        text_labels = []
        with open(self.txt_file, 'r') as file:
            for line in file:
                line = line.strip()
                content = line.split(',')
                file_path = content[0] 
                city_label = content[1] 
                state_label = content[2]
                country_label = content[3]
                continent_label = content[4]
                text_label = content[5:517] 
                scene_label = content[517:]
                file_path = file_path.strip()
                file_paths.append(file_path)
                city_labels.append(int(city_label.strip()))
                state_labels.append(int(state_label.strip()))
                country_labels.append(int(country_label.strip()))
                continent_labels.append(int(continent_label.strip()))
                text_labels.append([float(i.strip()) for i in text_label])
                scene_labels.append([float(i.strip()) for i in scene_label])
        return file_paths, city_labels, state_labels, country_labels, continent_labels, text_labels, scene_labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        city_label = self.city_labels[index]
        state_label = self.state_labels[index]
        country_label = self.country_labels[index]
        continent_label = self.continent_labels[index]
        text_label = self.text_labels[index]
        scene_label = self.scene_labels[index]

        frames = []
        frame_count = 0
        for frame_file in sorted(os.listdir(file_path)):
            if frame_count % 5 != 0:
                frame_path = os.path.join(file_path, frame_file)
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                frame = Image.fromarray(frame)  # Convert NumPy array to PIL image
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            frame_count += 1
            if len(frames) == 16:
                break
        frames_tensor = torch.stack(frames)  # Convert frames list to a tensor
        return frames_tensor, (city_label, state_label, country_label, continent_label, text_label, scene_label)
