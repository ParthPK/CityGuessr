import torch
import torch.nn as nn
from networks.VideoMAE import VideoMAE
from networks.classes import get_class_info
class_info = get_class_info()

class VanillaVideoTextScenes(nn.Module):
    def __init__(self):
        super(VanillaVideoTextScenes, self).__init__()
        
        self.head_count = 2
        self.embed_dim = 6
        self.n_scenes = 16
        self.text_dim = 512
        
        self.backbone = VideoMAE()
        self.n_features = self.backbone.n_features
        
        # Initialize classifiers
        self.classifier_continent = nn.Linear(self.n_features, class_info.num_labels_continent)
        self.classifier_country = nn.Linear(self.n_features, class_info.num_labels_country)
        self.classifier_state = nn.Linear(self.n_features, class_info.num_labels_state)
        self.classifier_city = nn.Linear(self.n_features, class_info.num_labels_city)
        
        self.query_projection = nn.Linear(1, self.embed_dim)
        self.key_projection = nn.Linear(1, self.embed_dim)
        self.value_projection = nn.Linear(1, self.embed_dim)
        self.final_projection = nn.Linear(self.embed_dim, 1)
        
        self.mha = nn.MultiheadAttention(self.embed_dim, self.head_count, batch_first=True)

        self.input_size = class_info.num_labels_continent + class_info.num_labels_country + class_info.num_labels_state + class_info.num_labels_city
        self.hidden_sizes = [256, 128, 64, 32, 16]
        self.out_linear = nn.Sequential(nn.Linear(self.input_size, self.hidden_sizes[0]),
        nn.ReLU(),
        nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
        nn.ReLU(),
        nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2]),
        nn.ReLU(),
        nn.Linear(self.hidden_sizes[2], self.hidden_sizes[3]),
        nn.ReLU(),
        nn.Linear(self.hidden_sizes[3], self.hidden_sizes[4]),
        nn.ReLU(),
        nn.Linear(self.hidden_sizes[4], self.n_scenes),
        #nn.Softmax(dim=1)
        )	
        
        self.text_linear = nn.Sequential(nn.Linear(self.input_size, 466),
        nn.ReLU(),
        nn.Linear(466, self.text_dim),
        #nn.Softmax(dim=1)
        )
	
    def forward(self, x):
        features = self.backbone(x)
        
        # Classify the features per hierarchy
        out_continent = self.classifier_continent(features)
        out_country = self.classifier_country(features)
        out_state = self.classifier_state(features)
        out_city = self.classifier_city(features)
        
        catted = torch.cat((out_city, out_state, out_country, out_continent), 1).unsqueeze(2)
        
        query = self.query_projection(catted)
        key = self.key_projection(catted)
        value = self.value_projection(catted)
        
        #add mha
        attended, attended_weights = self.mha(query, key, value)
        out_fin = self.final_projection(attended).squeeze(2)
        out_text = self.text_linear(out_fin)
        out_scene = self.out_linear(out_fin)
        
        out_list = [out_city, out_state, out_country, out_continent]
        
        return out_list, out_scene, out_text


if __name__ == "__main__":
    try:
        model = VanillaVideo()

        x = torch.rand(4, 16, 3, 224, 224)  # shape (batch_size, channels, frames, height, width)
        
        model, x = model.cuda(), x.cuda()
        
        output_list = model(x)
        
        print("Shapes of the outputs:")
        for i, out in enumerate(output_list):
            print(f"Output {i+1}: {out.shape}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
