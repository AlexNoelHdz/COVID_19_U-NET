import pickle
import torch
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.optim as optim
import torch.nn.functional as F

# =================================Diccionario de imagenes======================================== #

with open('S07_Exam_2/pickles/full_paths_dict.pickle', 'rb') as handle:
    full_paths_dict = pickle.load(handle)

from torch.nn import Module, Conv2d, Linear

class LungDataset(Dataset):
    def __init__(self, paths_dict, transform=None):
        self.data_dict = paths_dict
        self.transform = transform
        
    def __len__(self):
        return len(self.data_dict['images'])
    
    def __getitem__(self, idx):
        img_path = self.data_dict['images'][idx]
        mask_path = self.data_dict['lung masks'][idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

# Transformacion de las imagenes a tensores 256 x 256    
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = LungDataset(full_paths_dict['Train']['COVID-19'], transform=transform)
val_dataset = LungDataset(full_paths_dict['Val']['COVID-19'], transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# =================================Definicion del modelo======================================== #

class UnetModel(Module):
    def __init__(self) -> None:
        super().__init__()
        self.unet_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=3, out_channels=1, init_features=32, pretrained=True)
        
    def forward(self, x):
        return self.unet_model(x)

model = UnetModel()  # Asumiendo que ya has definido tu modelo U-Net
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 5

# Verifica si hay una GPU disponible y elige el dispositivo apropiado (GPU o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mueve tu modelo al dispositivo seleccionado
model.to(device)
losses = []  # Lista para almacenar las pérdidas

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, masks in train_loader:
       # Mueve las imágenes y máscaras al mismo dispositivo que el modelo
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = F.cross_entropy(output, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')