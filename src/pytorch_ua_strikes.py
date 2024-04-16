import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from util import Util
from sklearn.model_selection import train_test_split

#TODO
# - 3 different guesses but they're all very close to each other
# - test with more outputs ?
# - test loss function (euclid dist) on original network
# - try hierarchical method ?

util = Util()
util.read_data()

# Define neural network architecture
class MissileStrikePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MissileStrikePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Loss function for calculating distance between coordinates
def coordinate_distance_loss(predictions, targets):
    # Split predictions and targets into latitude and longitude
    pred_lat, pred_lon = predictions[:, 0], predictions[:, 1]
    target_lat, target_lon = targets[:, 0], targets[:, 1]
    
    # Calculate Euclidean distance between predicted and target coordinates
    distance = torch.sqrt((pred_lat - target_lat)**2 + (pred_lon - target_lon)**2)

    return distance.mean()

# Loss function that intakes multiple coordinates
def coordinates_average_distance_loss(predictions, targets):
    pred_lat, pred_lon = torch.chunk(predictions, 2, dim=1)
    target_lat, target_lon = torch.chunk(targets, 2, dim=1)
    
    # Calculate Euclidean distance between predicted and target coordinates
    distance = torch.sqrt((pred_lat - target_lat)**2 + (pred_lon - target_lon)**2)
    
    # Mask out loss contributions for padded coordinates
    mask = (target_lat != 0)  # Assuming 0 represents padding
    masked_distance = distance * mask.float()
    
    # Calculate the total number of valid coordinates (non-padded)
    num_valid_coordinates = torch.sum(mask.float())
    
    # Calculate the average distance, considering only valid coordinates
    average_distance = torch.sum(masked_distance) / num_valid_coordinates
    
    return average_distance

# Loss function that takes the 10%, 50%, and 90% quartile samples from 
# the differences between predictions and targets.
# ~4-5% increased accuracy compared to coordinates_average_distance_loss()
def spread_loss(predictions, targets):
    #disgusting 1-liner for changing the shape of the prediction tensor from 32x6 to 96x2
    # while keeping the gradient of the tensor.
    # OG format: [[lat1 lat2 lat3 long1 long2 long3], ...]
    # new format: [[lat1, long1], [lat2, long2], [lat3, long3], ...] 
    predictions = torch.stack([torch.stack((predictions[i][j],predictions[i][j+3])) \
                               for i in range(len(predictions)) for j in range(3)]).reshape(-1,2)
    abs_diff = torch.abs(predictions.view(-1, 3, 2) - targets.unsqueeze(1))
    quartiles = [0.1, 0.5, 0.9]
    losses = []

    for quartile in quartiles:
        quartile_loss = torch.quantile(abs_diff, quartile)
        losses.append(quartile_loss)

    total_loss = sum(losses)
    return total_loss


# Prepare data
data = util.get_data()
training_data = util.get_training_data()
testing_data = util.get_testing_data()
features = []
labels = []
for sample in training_data:
    features.append(sample[:3])
    labels.append(sample[3:])

# Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


# Convert data to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader for training and testing data
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize the model, loss function, and optimizer
input_size = 3  # Example: Number of input features (day of week, day of month, month of year)
hidden_size = 60
output_size = 2 * 3 # Output size for latitude and longitude
model = MissileStrikePredictor(input_size, hidden_size, output_size)
criterion = spread_loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for inputs, targets in test_loader:
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()

    test_loss /= len(test_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.4f}')


# Generate random input data for testing
num_samples = 100

while (num_samples > 0): 
    model.eval()

    # Create random values for day_of_week, day_of_month, and month
    day_of_week = torch.randint(low=0, high=7, size=(num_samples, 1), dtype=torch.float32)
    day_of_month = torch.randint(low=1, high=31, size=(num_samples, 1), dtype=torch.float32)
    month = torch.randint(low=1, high=13, size=(num_samples, 1), dtype=torch.float32)
    labels = []

    # Get correct labels for the data
    for i in range(num_samples):
        # Reformat into "weekday,day,month"
        date = ",".join([str(day_of_week[i].item())[:-2],str(day_of_month[i].item())[:-2],str(month[i].item())[:-2]])
        for entry in data: #find a sample for given date
            if (entry.startswith(date) and not (str(f"{i}:{entry}") in labels)):
                labels.append(f"{i}:{entry}")
                break



    # Concatenate the features along the second dimension to create x_test_tensor
    x_test_tensor = torch.cat((day_of_week, day_of_month, month), dim=1)

    # Make predictions
    with torch.no_grad():
        model_predictions = model(x_test_tensor)

    # If you want to convert the predictions to a numpy array
    predictions_np = model_predictions.numpy()

    losses = []
    passed = 0
    for i in range(len(labels)):
        label = labels[i].split(":")
        #separate label number & label data, then split coords into a float list
        index, target = int(label[0]), list(map(float,label[1].split(",")))[3:]
        distances = [[np.abs(predictions_np[index][i] - target[0]), np.abs(predictions_np[index][i + 3] - target[1])] \
                    for i in range(3)]
        loss = min([(coord[0] + coord[1]) / 2 for coord in distances])
        print(f"Loss: {loss}")
        if (loss < 1.8): passed += 1
        losses.append(loss)
        # print(f"{predictions_np[index]} | {labels[i]}\n\tBest Loss: {loss}")

    print("# of Samples:", len(labels))
    print("# of Passed:", passed)
    print(f"Accuracy: {100*passed/len(labels):0.3f}%")
    print(f"Average loss: {sum(losses)/len(labels):0.3f}")
    print(f"Lowest loss: {min(losses):0.3f}")
    print(f"Highest loss: {max(losses):0.3f}")

    num_samples = int(input("> ").strip())
