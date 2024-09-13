import torch
import torch.nn as nn
from transformers import VideoMAEModel

class VideoMAE(nn.Module):
    def __init__(self):
        super(VideoMAE, self).__init__()
        
        self.n_features = 384
        self.backbone = VideoMAEModel.from_pretrained("MCG-NJU/videomae-small-finetuned-kinetics")
        
        self.embeddings = self.backbone.embeddings
        self.encoder = self.backbone.encoder
        self.layernorm = self.backbone.layernorm if hasattr(self.backbone, 'layernorm') else None

    def encode_video(self, x):
        """ Stage #1: Get the embeddings of the video. """
        # Input:   torch.Tensor of shape (batch_size, n_frames, n_channels, width, height)
        # Returns: torch.Tensor of shape (batch_size, 1568, 768)
        return self.embeddings(x, None)

    def encode_embeddings(self, embedding_output, head_mask=None):
        """ Stage #2: Encode the video embeddings. """
        # Input:   torch.Tensor of shape (batch_size, 1568, 768)
        # Returns: torch.Tensor of shape (batch_size, 1568, 768)
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        sequence_output = encoder_outputs.last_hidden_state
        if self.layernorm is not None:
            sequence_output = self.layernorm(sequence_output)
        return sequence_output

    def get_last_hidden_state(self, x):
        """ Returns last hidden state from last layer. """
        # Returns: torch.Tensor of shape (batch_size, 1568, 768)
        return self.backbone(x, output_hidden_states=True).hidden_states[-1]

    def forward(self, x):
        """ Returns mean-pooled features from last layer. """
        # Returns: torch.Tensor of shape (batch_size, 768)
        return self.get_last_hidden_state(x).mean(1)


if __name__ == "__main__":
    try:
        # Initialize the VideoMAE model
        model = VideoMAE()

        # Generate some random video input data
        x = torch.rand(4, 16, 3, 224, 224)  # shape (batch_size, channels, frames, height, width)
        
        # Forward pass
        out = model(x)

        # Print shapes of the outputs
        print("Shape of the output from forward pass:", out.shape)
        
    except Exception as e:
        print(f"An error occurred: {e}")
