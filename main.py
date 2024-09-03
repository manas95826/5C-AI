import torch
import torch.nn as nn

class VNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNet, self).__init__()
        
        self.encoder1 = self.conv_block(in_channels, 16)
        self.encoder2 = self.conv_block(16, 32)
        self.encoder3 = self.conv_block(32, 64)
        self.encoder4 = self.conv_block(64, 128)
        
        self.bottleneck = self.conv_block(128, 256)
        
        self.decoder4 = self.upconv_block(256, 128)
        self.decoder3 = self.upconv_block(128, 64)
        self.decoder2 = self.upconv_block(64, 32)
        self.decoder1 = self.upconv_block(32, 16)
        
        self.final_conv = nn.Conv3d(16, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block
    
    def upconv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            self.conv_block(out_channels, out_channels)
        )
        return block
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        bottleneck = self.bottleneck(self.pool(enc4))
        
        dec4 = self.decoder4(bottleneck) + enc4
        dec3 = self.decoder3(dec4) + enc3
        dec2 = self.decoder2(dec3) + enc2
        dec1 = self.decoder1(dec2) + enc1
        
        return self.final_conv(dec1)
    
    def pool(self, x):
        return nn.MaxPool3d(kernel_size=2, stride=2)(x)

import torch.optim as optim

# Hyperparameters
learning_rate = 1e-4
epochs = 50

# Model, loss function, and optimizer
model = VNet(in_channels=1, out_channels=4)  # 4 output channels for the 4 organs
criterion = nn.CrossEntropyLoss()  # or Dice Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 10 == 9:
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 10}')
            running_loss = 0.0

print('Training completed.')

def dice_coefficient(pred, target):
    smooth = 1.0
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# Validation loop
model.eval()
dice_scores = {organ: [] for organ in ['Liver', 'Right Kidney', 'Left Kidney', 'Spleen']}

with torch.no_grad():
    for data in val_loader:
        inputs, labels = data
        outputs = model(inputs)
        
        for organ_idx, organ in enumerate(['Liver', 'Right Kidney', 'Left Kidney', 'Spleen']):
            dice_score = dice_coefficient(outputs[:, organ_idx, ...], labels[:, organ_idx, ...])
            dice_scores[organ].append(dice_score.item())
    
    for organ in dice_scores:
        print(f'{organ} Dice Score: {sum(dice_scores[organ]) / len(dice_scores[organ])}')

def predict(model, input_tensor):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        return torch.argmax(output, dim=1)

# Example inference
input_tensor = torch.randn(1, 1, 64, 64, 64)  # Example input shape
prediction = predict(model, input_tensor)

# 3-D Visualisation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

def plot_3d(image, threshold=-300):
    verts, faces, _, _ = measure.marching_cubes(image, threshold)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    ax.add_collection3d(mesh)
    plt.show()

plot_3d(prediction[0].cpu().numpy())
