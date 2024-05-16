'''
PyTorch version of neural network pair to predict missile strikes.

Example:
network = MissileStrikePredictor(train_data, test_data)
network.set_optimizer(optim.Adam(network.parameters(), lr=0.01))
network.train_model1(220)
date = [3, 23, 4]
network_pred = network.predict(date) (e.g. [45.35682, 24.59682])
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from util import Util
from sklearn.model_selection import train_test_split


util = Util()
util.read_data()
# Prepare data
data = util.get_data()
training_data = util.get_training_data()
testing_data = util.get_testing_data()

 
class RegionalMissileStrikePredictor(nn.Module):
    ''' Strike predictor neural network for a regional area '''
    def __init__(self, training_data, test_data):
        super(RegionalMissileStrikePredictor, self).__init__()
        self.training_data = training_data
        self.test_data = test_data
        self.train_loader = None
        self.test_loader = None

        self.optimizer = None
        self.criterion = self.spread_loss
        
        # Initialize the model, loss function, and optimizer
        input_size = 3  # Example: Number of input features (day of week, day of month, month of year)
        hidden_size = 110
        output_size = 2 * 3 # Output size for latitude and longitude (3 sets of coords)
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    # Loss function that takes the 10%, 50%, and 90% quartile samples from 
    # the differences between predictions and targets.
    # ~4-5% increased accuracy compared to coordinates_average_distance_loss()
    def spread_loss(self, predictions, targets):
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
    
    def prepare_data(self):
        features = []
        labels = []
        for sample in self.training_data:
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
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        self.test_loader = DataLoader(test_dataset, batch_size=32)

    def train_model(self, epochs=300):
        self.prepare_data()

        # Training loop
        for epoch in range(epochs):
            self.train()
            for inputs, targets in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            # Evaluation
            self.eval()
            with torch.no_grad():
                test_loss = 0.0
                for inputs, targets in self.test_loader:
                    outputs = self(inputs)
                    test_loss += self.criterion(outputs, targets).item()

            test_loss /= len(self.test_loader)
            print(f'Epoch {epoch+1}/{epochs}, Test Loss: {test_loss:.4f}')
    
    def convert_model1_output(self, coords):
        '''
        Return a region from the model1 output
        [lower-bound latitude, lower long, upper lat, upper long]
        e.g. [40.332, 34.565] -> [39.432, 33.515, 41.232, 35.615] 
        :param: coords - tuple (lat, long) 
        :param: radius - list [lat radius, long radius] 
        :return: list
        '''
        RADIUS = (1.1, 1.25) #for formatting model1 output

        return [coords[0]-RADIUS[0], coords[1]-RADIUS[1],
                coords[0]+RADIUS[0], coords[1]+RADIUS[1]]
    
    def predict(self, date):
        '''
        Create a prediction given a date
        :param: date (list[int]) - [day_of_week, day_of_month, month]
        :return: list, list - [lat, long], [region bounds from convert_model1_output]
        '''
        output = []
        with torch.no_grad():
            output = self(torch.tensor(date))
        output = [output[0], output[3]]

        return output, self.convert_model1_output(output)


class MissileStrikePredictor(nn.Module):
    ''' Strike predictor neural network '''
    def __init__(self, training_data, test_data):
        super(MissileStrikePredictor, self).__init__()
        self.training_data = training_data
        self.training_data_m2 = []
        self.test_data = test_data
        self.test_data_m2 = []
        self.train_loader = None
        self.test_loader = None

        self.optimizer = None
        self.criterion = self.coordinate_distance_loss

        input_size = 7  # Example: Number of input features (day of week, day of month, month of year)
        hidden_size = 14
        output_size = 2 # Output size for latitude and longitude
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.model1 = RegionalMissileStrikePredictor(training_data, test_data)
        self.optimizer_m1 = optim.Adam(self.model1.parameters(), lr=0.001)
        self.model1.set_optimizer(self.optimizer_m1)
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    # Loss function for calculating distance between coordinates
    def coordinate_distance_loss(self, predictions, targets):
        # Split predictions and targets into latitude and longitude
        pred_lat, pred_lon = predictions[:, 0], predictions[:, 1]
        target_lat, target_lon = targets[:, 0], targets[:, 1]
        
        # Calculate Euclidean distance between predicted and target coordinates
        distance = torch.sqrt((pred_lat - target_lat)**2 + (pred_lon - target_lon)**2)

        return distance.mean()

    # Loss function that intakes multiple coordinates
    def coordinates_average_distance_loss(self, predictions, targets):
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

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train_model1(self, epochs=300):
        self.model1.train_model(epochs)
    
    def prepare_data(self, m1_output):
        self.training_data_m2 = util.get_samples_in_range(self.training_data, m1_output)

        features_m2 = []
        labels_m2 = []
        #if training data is too small, m1 prediction was so bad that there aren't examples
        if (len(self.training_data_m2) < 3): return False
        for sample in self.training_data_m2:
            features_m2.append(sample[:3] + m1_output) #date + coordinate range
            labels_m2.append(sample[3:]) # target coordinates

        # Convert to numpy arrays
        features_m2 = np.array(features_m2)
        labels_m2 = np.array(labels_m2)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features_m2, labels_m2, test_size=0.25, random_state=42)


        # Convert data to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        # Create DataLoader for training and testing data
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        self.test_loader = DataLoader(test_dataset, batch_size=32)

        return True

    def print_losses(self, m_losses, m_passed, label_count):
        print("\n# of Samples:", label_count)
        print("# of Passed:", m_passed)
        print(f"Accuracy: {100*m_passed/label_count:0.3f}%")
        print(f"Average loss: {sum(m_losses)/label_count:0.3f}")
        print(f"Lowest loss: {min(m_losses) if m_losses else 0:0.3f}")
        print(f"Highest loss: {max(m_losses) if m_losses else 0:0.3f}")

    def train_model2(self, epochs=100):
        #train 2nd model 
        for epoch in range(epochs):
            self.train()
            for inputs, targets_m2 in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, targets_m2)
                loss.backward()
                self.optimizer.step()

            # # Evaluation
            # self.eval()
            # with torch.no_grad():
            #     test_loss = 0.0
            #     for inputs, targets_m2 in self.test_loader:
            #         outputs = self(inputs)
            #         test_loss += self.criterion(outputs, targets_m2).item()

            # test_loss /= len(self.test_loader)
            # print(f'Epoch {epoch+1}/{epochs}, Test Loss: {test_loss:.4f}')
        
    def predict(self, date, epochs=100):
        m1_output, m1_output_range = self.model1.predict(date)
        m2_input = date + m1_output_range
        #combine the coordinate range and date as input for evaluation 

        #get samples from M1 and load data into train loader  
        if (self.prepare_data(m1_output_range) == False):
            print("No good data")
            return [-1,-1]
        
        #train M2 using samples from within the region of M1's prediction
        self.train_model2(epochs=epochs)

        self.eval()
        # Make prediction
        with torch.no_grad():
            prediction = self(torch.tensor(m2_input))
        
        return prediction

    def test(self, num_samples=60):
        self.eval()

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
                    #format of labels: ["3:2,24,3,48.586,35.683", ...]
                    labels.append(f"{i}:{entry}")
                    break

        # Concatenate the features along the second dimension to create x_test_tensor
        x_test_tensor = torch.cat((day_of_week, day_of_month, month), dim=1)

        # Get prediction from Model 1
        with torch.no_grad():
            model_predictions = self.model1(x_test_tensor)

        #save list of coordinates from labels in float format
        targets = [list(map(float, label.split(":")[1].split(","))) for label in labels]

        # convert the predictions to a numpy array
        predictions = model_predictions.numpy()



util = Util()
util.read_data()
kyiv_range = [50.309243, 30.210609, 50.596237, 30.850794]
train_data = util.get_training_data()
# train_data = util.get_samples_in_range(train_data, kyiv_range)
test_data = util.get_testing_data()
# test_data = util.get_samples_in_range(test_data, kyiv_range)
model = MissileStrikePredictor(training_data=train_data, test_data=test_data)
criterion = model.coordinate_distance_loss
model.set_optimizer(optim.Adam(model.parameters(), lr=0.001))
model.train_model1(100000)

samples = 100
m2_epochs = 1000
count = 0
losses = []
while (samples > 0):
    passes = 0
    for i, sample in enumerate(test_data[:samples]):
        print(f"Sample: {i+1}/{samples}")
        date, target = sample[:3], sample[3:]

        model_pred = model.predict(date, m2_epochs)


        # get average loss of lat and long for both networks
        loss = (np.abs(model_pred[0]-target[0]) + \
                    np.abs(model_pred[1]-target[1]))/2

        loss_output = f"\tLoss: {loss:10.4f}"
        if (loss <= 0.5): 
            passes += 1
            loss_output += " ✔"
        losses.append(loss)
        print(loss_output)
        count += 1

    #if loop count is < samples, there weren't as many samples as requested
    if (count < samples): samples = count 
    print(f"Accuracy: {passes}/{samples} ({100*passes/samples:0.3f}%)")
    print(f"Average loss: {sum(losses)/len(losses):0.4f}")
    print(f"Minimum loss: {min(losses):0.4f}")
    print(f"Maximum loss: {max(losses):0.4f}")

    samples = int(input("S > ").strip())
    m2_epochs = int(input("E > ").strip())
