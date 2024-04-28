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

    def get_samples_in_range(self, coord_range):
        ''' Get all samples within a coordinate range
        :param: data - list of all samples
        :param: coord_range - list of lower & upper bounds for a coordinate
        :return: new_data - list of all samples within range of input coordinates 
        '''
        new_training_data = []
        for sample in self.training_data:
            #check if sample is within coordinate range
            if (coord_range[0] <= sample[3] <= coord_range[2] \
                and coord_range[1] <= sample[4] <= coord_range[3]):
                new_training_data.append(sample)
                
        return new_training_data
    
    def prepare_data(self, m1_output):
        self.training_data_m2 = self.get_samples_in_range(m1_output)

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
        
    def predict(self, date):
        m1_output, m1_output_range = self.model1.predict(date)
        m2_input = date + m1_output_range
        #combine the coordinate range and date as input for evaluation 

        #get samples from M1 and load data into train loader  
        if (self.prepare_data(m1_output_range) == False):
            print("No good data")
            return [-1,-1]
        
        #train M2 using samples from within the region of M1's prediction
        self.train_model2(epochs=450)

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




'''
model = MissileStrikePredictor(input_size_m1, hidden_size_m1, output_size_m1)
model2 = MissileStrikePredictor(input_size_m2, hidden_size_m2, output_size_m2)

criterion = model.spread_loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion_m2 = model2.coordinate_distance_loss
optimizer_m2 = optim.Adam(model2.parameters(), lr=0.01)

# Training loop
num_epochs = 250
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







def print_losses(m_losses, m_passed, label_count):
    print("\n# of Samples:", label_count)
    print("# of Passed:", m_passed)
    print(f"Accuracy: {100*m_passed/label_count:0.3f}%")
    print(f"Average loss: {sum(m_losses)/label_count:0.3f}")
    print(f"Lowest loss: {min(m_losses) if m_losses else 0:0.3f}")
    print(f"Highest loss: {max(m_losses) if m_losses else 0:0.3f}")



# Generate random input data for testing
num_samples = 60
run_m2 = 1 #flag for running results through 2nd model

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
                #format of labels: ["3:2,24,3,48.586,35.683", ...]
                labels.append(f"{i}:{entry}")
                break



    # Concatenate the features along the second dimension to create x_test_tensor
    x_test_tensor = torch.cat((day_of_week, day_of_month, month), dim=1)

    # Make predictions
    with torch.no_grad():
        model_predictions = model(x_test_tensor)

    #save list of coordinates from labels in float format
    targets = [list(map(float, label.split(":")[1].split(","))) for label in labels]

    # convert the predictions to a numpy array
    predictions_np = model_predictions.numpy()

    m1_losses = []
    m2_losses = []
    m1_passed = 0
    m2_passed = 0
    for sample_num, sample in enumerate(labels):
        print(f"Sample {sample_num+1}/{len(labels)}")
        #separate label number & label data, then split coords into a float list
        index, target = int(sample.split(":")[0]), targets[sample_num][3:]
        # distances format: [[0.39, 1.59], ...]
        distances = [[np.abs(predictions_np[index][i] - target[0]), np.abs(predictions_np[index][i + 3] - target[1])] \
                    for i in range(3)]

        # m1_loss_index, m1_loss = min(enumerate([(coord[0] + coord[1]) / 2 for coord in distances]), key=lambda x: x[1])
        m1_loss = (distances[0][0] + distances[0][1]) / 2
        m1_loss_index = 0
        print(f"\tM1 Loss: {m1_loss:0.4f}")
        if (m1_loss <= 1.8): m1_passed += 1
        m1_losses.append(m1_loss)
        
        if (run_m2):
            new_output = convert_model1_output([predictions_np[index][m1_loss_index], predictions_np[index][m1_loss_index+3]])
            training_data_m2, testing_data_m2 = get_samples_in_range(training_data, new_output)

            features_m2 = []
            labels_m2 = []
            if (len(training_data_m2) < 3): continue #if training data is too small, m1 prediction was so bad that there aren't examples
            for sample in training_data_m2:
                features_m2.append(sample[:3] + new_output) #date + coordinate range
                labels_m2.append(sample[3:]) # target coordinates

            # Convert to numpy arrays
            features_m2 = np.array(features_m2)
            labels_m2 = np.array(labels_m2)

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(features_m2, labels_m2, test_size=0.25, random_state=42)


            # Convert data to tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
            # X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            # y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

            # Create DataLoader for training and testing data
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            # test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            # test_loader = DataLoader(test_dataset, batch_size=32)

            
            #train 2nd model 
            num_epochs_m2 = 400
            for epoch in range(num_epochs_m2):
                model2.train()
                for inputs, targets_m2 in train_loader:
                    optimizer_m2.zero_grad()
                    outputs = model2(inputs)
                    loss = criterion_m2(outputs, targets_m2)
                    loss.backward()
                    optimizer_m2.step()

                # Evaluation
                # model.eval()
                # with torch.no_grad():
                #     test_loss = 0.0
                #     for inputs, targets_m2 in test_loader:
                #         outputs = model2(inputs)
                #         test_loss += criterion_m2(outputs, targets_m2).item()

                # test_loss /= len(test_loader)
                # print(f'Epoch {epoch+1}/{num_epochs_m2}, Test Loss: {test_loss:.4f}')
            
            #combine the coordinate range and date as input for evaluation 
            concat_tile = np.tile(new_output, (len(x_test_tensor), 1))
            x_test_tensor_m2 = torch.tensor(np.concatenate((x_test_tensor, concat_tile),axis=1), dtype=torch.float32)
            
            model2.eval()
            m2_loss = 0
            # Make predictions
            with torch.no_grad():
                #model2 returns a list of predictions. The sample_num corresponds to
                # each prediction
                model_predictions_m2 = model2(x_test_tensor_m2)
                # get average loss of lat and long
                m2_loss = (np.abs(model_predictions_m2[index][0]-target[0]) + \
                           np.abs(model_predictions_m2[index][1]-target[1]))/2
                print(f"mpm2: {len(model_predictions_m2)} | lm2: {len(labels_m2)}")

            # print(f"Prediction: {model_predictions_m2[sample_num]} | Actual: {target})")
            #check if loss is within bound (~62km)
            if (m2_loss <= 0.5): m2_passed += 1
            m2_losses.append(m2_loss)
            print(f"\t\tM2 Loss: {m2_loss:0.4f}")
        

    
    print_losses(m1_losses, m1_passed, len(labels))
    print_losses(m2_losses, m2_passed, len(labels))


    num_samples = int(input("> ").strip())


'''


'''
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
'''
