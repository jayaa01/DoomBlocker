import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. Custom Dataset Loader for .npy Sequence Files
class DoomDataset(Dataset):
    def __init__(self, data_dir="data"):
        self.samples = []
        self.labels = []
        self.class_map = {"studying": 0, "scrolling": 1, "idle": 2}

        for class_name, label in self.class_map.items():
            folder = os.path.join(data_dir, class_name)
            files = glob.glob(os.path.join(folder, "*.npy"))
            print(f"Loaded {len(files)} samples for class '{class_name}'")
            for f in files:
                seq = np.load(f)  # Shape: (30, 4)
                self.samples.append(seq)
                self.labels.append(label)

        self.samples = np.array(self.samples, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Transpose shape from (30, 4) to (4, 30) for 1D Convolution
        x = torch.tensor(self.samples[idx]).T
        y = torch.tensor(self.labels[idx])
        return x, y

# 2. Lightweight 1D-CNN Sequence Classifier
class ActionClassifier1D(nn.Module):
    def __init__(self, num_classes=3):
        super(ActionClassifier1D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # Shape: (16, 15)
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),      # Shape: (32, 1)
            
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 3. Training Loop
def main():
    os.makedirs("models", exist_ok=True)
    dataset = DoomDataset("data")

    if len(dataset) == 0:
        print("[!] Error: No data found in 'data/' directory. Make sure you recorded sequences first.")
        return

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = ActionClassifier1D(num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nStarting Training Pipeline...")
    EPOCHS = 35

    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        acc = (correct / total) * 100
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {total_loss/len(dataloader):.4f} | Accuracy: {acc:.2f}%")

    # Save Model Weights
    save_path = os.path.join("models", "action_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\n[✔] Model successfully trained and saved to '{save_path}'!")

if __name__ == "__main__":
    main()
