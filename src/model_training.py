import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

# from maxvit import MaxViT
from multihead import ResNetWithAttentionAndSqueeze


# model = MaxViT(
#     num_classes = 2,
#     dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
#     dim = 96,                         # dimension of first layer, doubles every layer
#     dim_head = 32,                    # dimension of attention heads, kept at 32 in paper
#     depth = (2, 2, 5, 2),             # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
#     window_size = 7,                  # window size for block and grids
#     mbconv_expansion_rate = 4,        # expansion rate of MBConv
#     mbconv_shrinkage_rate = 0.25,     # shrinkage rate of squeeze-excitation in MBConv
#     dropout = 0.1                     # dropout
# )

model = ResNetWithAttentionAndSqueeze()

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self._load_images()
        self.transform = transform
        
    def _load_images(self):
        images = []
        for cls in self.classes:
            class_path = os.path.join(self.root_dir, cls)
            for image_name in os.listdir(class_path):
                images.append((os.path.join(class_path, image_name), cls))
        return images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, cls = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        label = self.class_to_idx[cls]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label


transform = transforms.Compose([transforms.Resize((36, 6000)),
                                transforms.ToTensor()])


train_data = CustomDataset(r'proper_dataset\train', transform)
test_data= CustomDataset(r'proper_dataset\test', transform)

train_loader = DataLoader(dataset=train_data, batch_size=2, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=2, shuffle=True)

# Trainig Setting
lr = 3e-5
gamma = 0.7
# seed = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 100
# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
model.to(device)
train_losses = []
val_losses = []
test_losses = []

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    # Initialize tqdm for the training loop
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='batch') as train_pbar:
        for i, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc.item()  # Accumulate accuracy
            epoch_loss += loss.item()  # Accumulate loss

            # Update tqdm progress bar with current loss and accuracy
            train_pbar.set_postfix(loss=loss.item(), acc=acc.item())
            train_pbar.update()

    # Calculate average accuracy and loss for the epoch
    epoch_accuracy /= len(train_loader)
    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)
  
    # Testing loop with tqdm
    epoch_test_accuracy = 0
    epoch_test_loss = 0
    predictions = []
    true_labels = []


    with tqdm(total=len(test_loader), desc=f'testidation {epoch}/{epochs}', unit='batch') as test_pbar:
        for j, (test_data, test_label) in enumerate(test_loader):
            test_data = test_data.to(device)
            test_label = test_label.to(device)

            test_output = model(test_data)
            test_loss = criterion(test_output, test_label)

            acc = (test_output.argmax(dim=1) == test_label).float().mean()
            epoch_test_accuracy += acc.item()  # Accumulate testidation accuracy
            epoch_test_loss += test_loss.item()  # Accumulate testidation loss
            predictions.extend(test_output.argmax(dim=1).cpu().detach().numpy())
            true_labels.extend(test_label.cpu().detach().numpy())

            # Update tqdm progress bar for testidation
            test_pbar.set_postfix(loss=test_loss.item(), acc=acc.item())
            test_pbar.update()

    # Calculate average test accuracy and loss for the epoch
    epoch_test_accuracy /= len(test_loader)
    epoch_test_loss /= len(test_loader)
    test_losses.append(epoch_test_loss)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    conf_matrix = confusion_matrix(true_labels, predictions)

    # Save precision, recall, F1-score, and confusion matrix to a CSV file
    results_dict = {
        'Precision': [precision],
        'Recall': [recall],
        'F1-Score': [f1]
    }
    conf_matrix_df = pd.DataFrame(conf_matrix)
    conf_matrix_df.to_csv(f'results/confusion_matrix_epoch_{epoch}.csv', index=False)
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(f'results/evaluation_metrics_epoch_{epoch}.csv', index=False)
    

    # # Print and save model after each epoch
    print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - test_loss : {epoch_test_loss:.4f} - test_acc: {epoch_test_accuracy:.4f}\n")
    torch.save(model.state_dict(), f'models/pytorch_epoch_{epoch}.pb')
# Plot the losses
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('loss')
