# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Image classification is a fundamental task in computer vision where an input image is assigned to one of several predefined classes. The objective of this experiment is to build and train a Convolutional Neural Network (CNN) using a labeled image dataset and evaluate its performance using accuracy, confusion matrix, and classification report.

## Neural Network Model
<img width="1300" height="657" alt="567661993-26a0badc-b171-4b91-8762-59a0643e06bd" src="https://github.com/user-attachments/assets/444036a9-a81a-42fa-b699-0d6e2cd630c1" />


## DESIGN STEPS
### STEP 1: 
Import the required libraries (torch, torchvision, torch.nn, torch.optim) and load the image dataset with necessary preprocessing like normalization and transformation.

### STEP 2: 

Split the dataset into training and testing sets and create DataLoader objects to feed images in batches to the CNN model.

### STEP 3: 

Define the CNN architecture using convolutional layers, ReLU activation, max pooling layers, and fully connected layers as implemented in the CNNClassifier class.

### STEP 4: 
Initialize the model, define the loss function (CrossEntropyLoss), and choose the optimizer (Adam) for training the network.


### STEP 5: 

Train the model using the training dataset by performing forward pass, computing loss, backpropagation, and updating weights for multiple epochs.

### STEP 6: 

Evaluate the trained model on test images and verify the classification accuracy for new unseen images.



## PROGRAM

### Name: R.Bharathkumar

### Register Number: 212224103001

```python
class CNNClassifier(nn.Module):
   def __init__(self, input_size):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(128*3*3,128) # Assuming input_size leads to 3x3 after pooling
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)

   def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0),-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier(28)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model
def train_model(model, train_loader, num_epochs=3):
for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Name:R.Bharathkumar')
        print('Register Number: 212224103001')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```

### OUTPUT

## Training Loss per Epoch

<img width="546" height="300" alt="Screenshot 2026-05-18 133313" src="https://github.com/user-attachments/assets/afab1d6a-f58f-41b8-b545-ea74ab1b1b42" />


## Confusion Matrix


<img width="928" height="757" alt="Screenshot 2026-05-18 133324" src="https://github.com/user-attachments/assets/897f85b6-256e-4d36-898e-b4ca4d725f61" />


## Classification Report
<img width="562" height="436" alt="Screenshot 2026-05-18 133337" src="https://github.com/user-attachments/assets/2220299c-1b9c-45d3-9c82-ee985cf55c2d" />



### New Sample Data Prediction
<img width="621" height="637" alt="Screenshot 2026-05-18 133409" src="https://github.com/user-attachments/assets/1d3a7115-7e6b-48f7-b6de-2532e705cf51" />


## RESULT
Thus, To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images is executed and verified successfully.
