import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import ViTModel, T5Tokenizer, T5ForConditionalGeneration, AutoImageProcessor
from PIL import Image


class LayerNorm(nn.Module):
    """T5-style LayerNorm over the channel dimension (No bias and no subtraction of mean)."""
    def __init__(self, n_channels):
        super(LayerNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(n_channels, 1, 1))

    def forward(self, x: torch.Tensor):
        # x is a feature map of shape: batch_size x n_channels x h x w
        var = x.square().mean(dim=1, keepdim=True)
        out = x * (var + 1e-8).rsqrt()
        out = out * self.scale
        return out


class HeatmapPredictor(nn.Module):
    def __init__(self, n_channels):
        super(HeatmapPredictor, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_channels, 768, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            LayerNorm(768),
            nn.Conv2d(768, 384, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            LayerNorm(384),
        )

        self.deconv_layers = nn.ModuleList()
        self.conv_layers2 = nn.ModuleList()
        in_channels = 384
        for out_channels in [768, 384, 384, 192]:
            self.deconv_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                    LayerNorm(out_channels),
                    nn.ReLU()
                )
            )
            self.conv_layers2.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'),
                    LayerNorm(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'),
                    LayerNorm(out_channels),
                )
            )
            in_channels = out_channels

        self.relu = nn.ReLU()

        self.last_conv1 = nn.Conv2d(in_channels, 192, kernel_size=3, stride=1, padding='same')
        # relu
        self.last_conv2 = nn.Conv2d(192, 1, kernel_size=3, stride=1, padding='same') # final_channel size
        # sigmoid
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv_layers(x)
        for deconv, conv in zip(self.deconv_layers, self.conv_layers2):
            x = deconv(x)
            identity = x
            x = conv(x)
            x = x + identity
            x = self.relu(x)

        x = self.last_conv1(x)
        x = self.relu(x)
        x = self.last_conv2(x)
        x = self.sigmoid(x)  # (batch_size, 1, height, width)

        output = x.squeeze(1)
        return output


class ScorePredictor(nn.Module):
    def __init__(self, n_channels, n_patches=14*14):
        super(ScorePredictor, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1),
            LayerNorm(n_channels),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels // 2, kernel_size=3, stride=1, padding=1),
            LayerNorm(n_channels // 2),
            nn.ReLU(),
            nn.Conv2d(n_channels // 2, n_channels // 4, kernel_size=3, stride=1, padding=1),
            LayerNorm(n_channels // 4),
            nn.ReLU(),
            nn.Conv2d(n_channels // 4, n_channels // 8, kernel_size=3, stride=1, padding=1),
            LayerNorm(n_channels // 8),
            nn.ReLU(),
            nn.Conv2d(n_channels // 8, n_channels // 16, kernel_size=3, stride=1, padding=1),
            LayerNorm(n_channels // 16),
            nn.ReLU(),
            nn.Conv2d(n_channels // 16, n_channels // 64, kernel_size=3, stride=1, padding=1),
            LayerNorm(n_channels // 64),
            nn.ReLU(),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(n_channels // 64 * n_patches, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        conv_output = self.conv_layers(x)
        conv_output = conv_output.flatten(1)
        output = self.linear_layers(conv_output)
        return output


class RAHF(nn.Module):
    def __init__(
            self,
            score_types=('plausibility', 'alignment', 'aesthetics', 'overall'),
            heatmap_types=('implausibility', 'misalignment'),
            vit_model="google/vit-large-patch16-384",
            t5_model="t5-base",
            multi_heads=True,
            patch_size=16,
            image_size=384,
        ):
        super(RAHF, self).__init__()
        self.multi_heads = multi_heads
        self.score_types = score_types
        self.heatmap_types = heatmap_types
        self.n_patches = image_size // patch_size

        # Load pre-trained ViT model for image encoding
        self.vit = ViTModel.from_pretrained(vit_model)

        # Load pre-trained T5 model
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model)
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model)

        # Linear layer to align visual token dimensions to T5 hidden size
        self.visual_token_projection = nn.Linear(self.vit.config.hidden_size, self.t5.config.d_model)

        n_channels = self.t5.config.d_model
        if self.multi_heads:
            self.heatmap_predictor = nn.ModuleDict({hm: HeatmapPredictor(n_channels) for hm in heatmap_types})
            self.score_predictor = nn.ModuleDict({score: ScorePredictor(n_channels, self.n_patches ** 2) for score in score_types})
        else:
            # Single head
            self.heatmap_predictor = HeatmapPredictor(n_channels)
            self.score_predictor = ScorePredictor(n_channels, self.n_patches ** 2)

    def encode_vis_text(self, visual_tokens, caption, prepend_text=None):
        if prepend_text is not None:
            caption = [f"<output> {prepend_text} </output> {c}" for c in caption]
        # Tokenize the caption
        input_ids = self.tokenizer(caption, return_tensors="pt", padding=True, truncation=True).input_ids.to(visual_tokens.device)

        # Embed textual tokens using T5's embedding layer
        textual_embeddings = self.t5.encoder.embed_tokens(input_ids)  # (batch_size, seq_len, t5_hidden_dim)

        # Concatenate visual tokens and textual embeddings along the sequence length dimension
        concatenated_tokens = torch.cat([visual_tokens, textual_embeddings], dim=1)  # (batch_size, seq_len + num_patches, t5_hidden_dim)

        # Encode the concatenated tokens with T5 encoder
        encoder_outputs = self.t5.encoder(inputs_embeds=concatenated_tokens)

        if prepend_text is not None:
            # Extract visual tokens from encoder outputs (remove CLS token)
            visual_tokens = encoder_outputs.last_hidden_state[:, 1:visual_tokens.size(1), :]  # Visual tokens portion
            batch_size = visual_tokens.shape[0]
            feature_map = visual_tokens.transpose(1, 2).view(batch_size, -1, self.n_patches, self.n_patches)  # Reshape to (batch_size, t5_hidden_size, height, width)
            return feature_map
        
        return encoder_outputs

    def forward(self, image, caption, target_text=None, max_new_tokens=100):
        # Encode the image using ViT
        vit_outputs = self.vit(pixel_values=image)
        visual_tokens = vit_outputs.last_hidden_state  # (batch_size, num_patches, vit_hidden_dim)
        batch_size = visual_tokens.shape[0]

        # Project visual tokens to T5 hidden size
        visual_tokens = self.visual_token_projection(visual_tokens)  # (batch_size, num_patches, t5_hidden_dim)

        encoder_outputs = self.encode_vis_text(visual_tokens, caption)

        last_hidden_state = encoder_outputs.last_hidden_state
        encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state[:, visual_tokens.size(1):]

        outputs = {}

        # Compute outputs using T5's decoder with a generation head
        if target_text is not None:
            # Prepare decoder inputs for teacher forcing
            caption_inputs = self.tokenizer(target_text, return_tensors="pt", padding=True, truncation=True).to(image.device)
            decoder_input_ids = caption_inputs.input_ids

            t5_outputs = self.t5(
                encoder_outputs=encoder_outputs,
                labels=decoder_input_ids  # Using labels computes the loss automatically
            )

            loss = t5_outputs.loss  # Teacher-forcing loss is automatically computed
            # logits = t5_outputs.logits  # Predicted vocabulary scores
            outputs['seq_loss'] = loss
        else:
            # Generation texts
            output_seq = self.t5.generate(
                encoder_outputs=encoder_outputs,
                max_new_tokens=max_new_tokens,
            )
            pred_seq = self.tokenizer.batch_decode(output_seq, skip_special_tokens=True)
            outputs['output_seq'] = pred_seq

        if self.multi_heads:
            # Extract visual tokens from encoder outputs (remove CLS token)
            visual_tokens = last_hidden_state[:, 1:visual_tokens.size(1), :]  # Visual tokens portion
            feature_map = visual_tokens.transpose(1, 2).view(batch_size, -1, self.n_patches, self.n_patches)  # Reshape to (batch_size, t5_hidden_size, height, width)

            heatmaps = {hm: hmp(feature_map) for hm, hmp in self.heatmap_predictor.items()}
            scores = {sc: scp(feature_map).flatten() for sc, scp in self.score_predictor.items()}
        else:
            scores = {}
            for score in self.score_types:
                feature_map = self.encode_vis_text(visual_tokens, caption, prepend_text=f'SCORE: {score}')
                scores[score] = self.score_predictor(feature_map).flatten()

            heatmaps = {}
            for heatmap in self.heatmap_types:
                feature_map = self.encode_vis_text(visual_tokens, caption, prepend_text=f'HEATMAP: {heatmap}')
                heatmaps[heatmap] = self.heatmap_predictor(feature_map)

        outputs['heatmaps'] = heatmaps
        outputs['scores'] = scores

        return outputs


def preprocess_image(image_path):
    transform = AutoImageProcessor.from_pretrained("google/vit-large-patch16-384")
    image = Image.open(image_path).convert("RGB")
    return transform(image, return_tensors="pt")['pixel_values'][0].unsqueeze(0)


if __name__ == "__main__":
    # Example usage
    image_path = "data/a.jpg"
    caption = "A description of the image"

    image_tensor = preprocess_image(image_path)
    model = RAHF()

    out = model(image_tensor, caption, target_text=None)
