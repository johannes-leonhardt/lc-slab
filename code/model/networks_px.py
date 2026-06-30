import torch
import torch.nn as nn
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import transformers

## Own network implementations

class CNN(nn.Module):

    def __init__(self, n_in, n_out, n_hidden, n_layers):

        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.n_layer = n_layers

        self.input_layer = _SingleLinear(n_in, n_hidden)
        self.hidden_layers = nn.ModuleList([_SingleConv(n_hidden, n_hidden) for _ in range(n_layers)])
        self.output_layer = nn.Conv2d(n_hidden, n_out, kernel_size=1)

    def forward(self, x):

        x = x[0]
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)

        return x, None
    
SimpleCNN = CNN # Alias
    
class _SingleLinear(nn.Module):

    def __init__(self, n_in, n_out):

        super().__init__()

        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x
    
class _SingleConv(nn.Module):

    def __init__(self, n_in, n_out):

        super().__init__()

        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x
    
# SMP implementations
    
class UNet_SMP(nn.Module):

    def __init__(self, n_in, n_out, n_layers, encoder, upscale=True):

        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_layer = n_layers
        self.encoder = encoder

        self.upscale = transforms.Resize((224, 224)) if upscale else nn.Identity()
        self.network = smp.Unet(encoder, encoder_weights=None, in_channels=n_in, classes=64)
        self.downscale = nn.AdaptiveAvgPool2d((64, 64))
        self.output_layer = nn.Conv2d(64, n_out, kernel_size=1)

    def forward(self, x):

        x = x[0]
        x = self.upscale(x)
        x = self.network(x)
        x = self.downscale(x)
        out = self.output_layer(x)

        return out, x
    
class UNetPP_SMP(nn.Module):

    def __init__(self, n_in, n_out, n_layers, encoder, upscale=True):

        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_layer = n_layers
        self.encoder = encoder
        
        self.upscale = transforms.Resize((224, 224)) if upscale else nn.Identity()
        self.network = smp.UnetPlusPlus(encoder, encoder_weights=None, in_channels=n_in, classes=64)
        self.downscale = nn.AdaptiveAvgPool2d((64, 64))
        self.output_layer = nn.Conv2d(64, n_out, kernel_size=1)

    def forward(self, x):

        x = x[0]
        x = self.upscale(x)
        x = self.network(x)
        x = self.downscale(x)
        out = self.output_layer(x)

        return out, x
    
class DeepLabV3_SMP(nn.Module):

    def __init__(self, n_in, n_out, n_layers, encoder, upscale=True):

        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_layer = n_layers
        self.encoder = encoder
        
        self.upscale = transforms.Resize((224, 224)) if upscale else nn.Identity()
        self.network = smp.DeepLabV3Plus(encoder, encoder_weights=None, in_channels=n_in, classes=64)
        self.downscale = nn.AdaptiveAvgPool2d((64, 64))
        self.output_layer = nn.Conv2d(64, n_out, kernel_size=1)

    def forward(self, x):
        
        x = x[0]
        x = self.upscale(x)
        x = self.network(x)
        x = self.downscale(x)
        out = self.output_layer(x)

        return out, x
    
class UPerNet_SMP(nn.Module):

    def __init__(self, n_in, n_out, n_layers, encoder, upscale=True):

        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_layer = n_layers
        self.encoder = encoder
        
        self.upscale = transforms.Resize((224, 224)) if upscale else nn.Identity()
        self.network = smp.UPerNet(encoder, encoder_weights=None, in_channels=n_in, classes=64)
        self.downscale = nn.AdaptiveAvgPool2d((64, 64))
        self.output_layer = nn.Conv2d(64, n_out, kernel_size=1)

    def forward(self, x):
        
        x = x[0]
        x = self.upscale(x)
        x = self.network(x)
        x = self.downscale(x)
        out = self.output_layer(x)

        return out, x
    
# Huggingface implementations

class Segformer_HF(nn.Module):

    def __init__(self, n_in, n_out, encoder, upscale=True):

        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.encoder = encoder
        
        self.upscale = transforms.Resize((224, 224)) if upscale else nn.Identity()
        if encoder == "mitb0":
            config = transformers.SegformerConfig(num_channels=n_in, num_labels=n_out)
        elif encoder == "mitb1":
            config = transformers.SegformerConfig(num_channels=n_in, num_labels=n_out, depths=[2,2,2,2], hidden_sizes=[64,128,320,512], decoder_hidden_size=256)
        elif encoder == "mitb2":
            config = transformers.SegformerConfig(num_channels=n_in, num_labels=n_out, depths=[3,4,6,3], hidden_sizes=[64,128,320,512], decoder_hidden_size=768)
        elif encoder == "mitb3":
            config = transformers.SegformerConfig(num_channels=n_in, num_labels=n_out, depths=[3,4,18,3], hidden_sizes=[64,128,320,512], decoder_hidden_size=768)
        elif encoder == "mitb4":
            config = transformers.SegformerConfig(num_channels=n_in, num_labels=n_out, depths=[3,4,27,3], hidden_sizes=[64,128,320,512], decoder_hidden_size=768)
        elif encoder == "mitb5":
            config = transformers.SegformerConfig(num_channels=n_in, num_labels=n_out, depths=[3,6,40,3], hidden_sizes=[64,128,320,512], decoder_hidden_size=768)
        self.network = transformers.SegformerForSemanticSegmentation(config)
        self.downscale = nn.AdaptiveAvgPool2d((64, 64))

    def forward(self, x):

        x = x[0]
        x = self.upscale(x)
        x = self.network(x, output_hidden_states=True, return_dict=True)
        feats = x.hidden_states
        feats = torch.cat([self.downscale(feats_i) for feats_i in feats], dim=1)
        out = x.logits
        out = self.downscale(out)

        return out, feats
