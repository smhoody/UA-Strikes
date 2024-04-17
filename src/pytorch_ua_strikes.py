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
num_epochs = 400
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


def relu(z): #Rectified Linear Unit activation function
    return np.maximum(0, z)

def deriv_relu(z): #ReLU derivative
    return (z > 0).astype(float)

def convert_model1_output(coords):
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


def get_samples(data, coord_range):
    ''' Get all samples within a coordinate range
    :param: data - list of all samples
    :param: coord_range - list of lower & upper bounds for a coordinate
    :return: new_data - list of all samples within range of input coordinates 
    '''
    new_training_data = []
    new_testing_data = []
    counter = 0
    for sample in data:
        #check if sample is within coordinate range
        if (coord_range[0] <= sample[3] <= coord_range[2] \
            and coord_range[1] <= sample[4] <= coord_range[3]):
            
            counter+=1
            if (counter % 4 == 0):
                new_testing_data.append(sample)
            else:
                new_training_data.append(sample)
            
    return new_training_data, new_testing_data


#Model 2 functions

def train_model2(data, model1):
    ''' Train the 4x3x2 model with given data '''
    n = 0.00001 # learning rate

    #4x3 input layer weight matrix
    # 67.742% test sample | 19.923% over 516 samples
    w0 = np.array([[-0.89831668, -1.19550875, -0.02574357],
                   [-1.8221221,  -0.64135006, -0.01782699],
                   [ 1.57536582,  0.60907062, -0.02687357],
                   [ 0.98955667,  1.40979433, -0.01912199]])
    
    #3x2 layer 1 weight matrix
    # 67.742%
    w1 = np.array([[2.08848719, 1.45993777],
                   [2.84632471, 2.07641199],
                   [0.00770026, 0.00553103]])

    epochs = 3000
    bias_1 = 3.656516218491046 #layer 1 bias (67.742%)
    bias_2 = 2.636866698491047 #layer 2 bias (67.742%)
    loss = 0
    samples = len(data)

    for epoch in range(epochs):
        epoch_loss = 0
        
        for sample_num, sample in enumerate(data):
            # sample = sample.split(",")[3:]
            # print(sample[:3])
            # prediction = predict_model1(sample[:3], m1_w0, m1_w1, m1_b1, m1_b2) #get prediction from model1
            m1_input = torch.stack(tuple(torch.tensor(sample[:3])))
            # Make predictions
            with torch.no_grad():
                m1_prediction = model1(m1_input)
            input = convert_model1_output([m1_prediction[0], m1_prediction[3]])#get coordinate range
            target = sample[3:] #get lat,long 
            
            #---Forward prop---
            #calculate z = W_T*x + b for layer 1
            weighted_sum = np.dot(w0.T, input) #multiply weights x input vector
            z1 = weighted_sum + bias_1

            #pass layer 0 sum to activation function
            yHat_l0 = relu(z1)

            #calculate z = W_T*x + b for layer 2
            weighted_sum = np.dot(w1.T, yHat_l0)
            z2 = weighted_sum + bias_2

            #pass layer 1 sum to activation function
            yHat_l1 = relu(z2)

            loss = 0.5 * np.power(yHat_l1 - target, 2)
            epoch_loss += loss

            #---Backpropagation---
            # dE/dPredict
            loss_deriv = np.subtract(yHat_l1, target) #2x1 vector
            # dPredict/dReLU for layer 1 and 2
            relu_deriv_l2 = deriv_relu(z2) #2x1 vector 
            relu_deriv_l1 = deriv_relu(z1) #3x1 vector

            #calculate dL/dWi = dL*dReLU * Wi   (i=layer)
            #first take Hadamard product of Loss' & ReLU' 
            loss_relu_dot_prod2 = np.multiply(loss_deriv, relu_deriv_l2) #2x1 vector
            loss_relu_dot_prod1 = np.multiply(np.append(loss_deriv,1), relu_deriv_l1) #3x2 vector

            #use dot product to adjust bias   dC/dB = dC/dZ
            bias_2 = bias_2 - n*np.dot(loss_deriv, relu_deriv_l2) #scalar
            bias_1 = bias_1 - n*np.dot(np.append(loss_deriv,1), relu_deriv_l1) #scalar
            #append a 0 to loss_deriv in order to project it onto a 3D space to get dot prod

            # calculate gradient of L W.R.T weights in input layer
            #dL/dw0 = d/dw0(input*w0) = input
            partial_L_w0 = np.outer(loss_relu_dot_prod1, input)

            # calculate gradient of L W.R.T weights in layer 1
            #dL/dw1 = d/dw1(yHat_l0*w1) = yHat_l0
            partial_L_w1 = np.outer(loss_relu_dot_prod2, yHat_l0)

            # update weight matricies for layer 0 and 1
            # w = w - a * dE/dw
            w0 = w0 - (n * partial_L_w0).T
            w1 = w1 - (n * partial_L_w1).T
            
        # print(f"Epoch {epoch+1} loss: {epoch_loss/samples}")

    return w0, w1, bias_1, bias_2 


def test_model2(w0, w1, bias_1, bias_2, m1_w0, m1_w1, m1_b1, m1_b2, data):
    ''' Test 4x3x2 model (precision model) with unseen data '''
    passed = 0
    total = len(data)
    correct_coords = {}
    losses = []

    for sample_num, sample in enumerate(data):
        prediction = model(torch.cat(tuple(sample[:3])))
        print("model1 output:")
        print(prediction)
        input = convert_model1_output(prediction)#get coordinate range
        target = sample[3:] #get lat,long 

        weighted_sum = np.dot(w0.T, input) #multiply weights x input vector
        z1 = np.add(weighted_sum, bias_1)
        #pass layer 0 sum to activation function
        yHat_l0 = relu(z1)

        #calculate z = W_T*x + b for layer 2
        weighted_sum = np.dot(w1.T, yHat_l0)
        z2 = np.add(weighted_sum, bias_2)

        #pass layer 1 sum to activation function
        yHat_l1 = relu(z2)

        loss = np.absolute(np.subtract(yHat_l1, target))
        # print(f"Prediction: {yHat_l1}  Target: {target}  |  loss: {loss}")
        #correct approx if loss is within 0.45 deg latitude & 0.525 deg longitude (~57km^2)
        if ((loss < [0.45, 0.52]).all()): 
            passed += 1
            correct_coords[",".join(map(str,input))] = yHat_l1

        losses.append(loss)


    print(f"\nTotal permissible: {passed}")
    print(f"Accuracy: {passed/total*100:0.3f}%\n")



def predict_model2(date, w0, w1, b1, b2):
    ''' Create a prediction using the Model 2 (precision) network 
    :return: list  [latitude, longitude]
    '''
    #input layer
    weighted_sum = np.dot(w0.T, date)
    z1 = weighted_sum + b1
    yHat_l0 = relu(z1)

    #layer 1
    weighted_sum = np.dot(w1.T, yHat_l0)
    z2 = weighted_sum + b2
    yHat_l1 = relu(z2) #final answer

    return yHat_l1


# Generate random input data for testing
num_samples = 40

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
                #format of labels: ["3:2,24,3,48.586,35.683", ...]
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
        print(f"Sample {sample_num}/{len(labels)}")
        #separate label number & label data, then split coords into a float list
        index, target = int(sample.split(":")[0]), targets[sample_num][3:]
        distances = [[np.abs(predictions_np[index][i] - target[0]), np.abs(predictions_np[index][i + 3] - target[1])] \
                    for i in range(3)]
        #TODO
        # change m1_loss to be the first pair so that it matches 
        # whats in m2training. 
        # Do few samples but 2000-3000 epochs for m2
        # m1_loss_index, m1_loss = min(enumerate([(coord[0] + coord[1]) / 2 for coord in distances]), key=lambda x: x[1])
        m1_loss = (distances[0][0] + distances[0][1]) / 2
        m1_loss_index = 0
        print(f"M1 Loss: {m1_loss}")
        if (m1_loss < 1.8): m1_passed += 1
        m1_losses.append(m1_loss)
        
        new_output = convert_model1_output([predictions_np[index][m1_loss_index], predictions_np[index][m1_loss_index+3]])
        # print(new_output)
        
        model2_training_data, model2_testing_data = get_samples(testing_data, new_output)

        #train 2nd model (4x3x2)
        m2_w0, m2_w1, m2_b1, m2_b2 = train_model2(model2_training_data, model)
        # test_model2(m2_w0, m2_w1, m2_b1, m2_b2, w0, w1, b1, b2, model2_training_data + model2_testing_data)
        prediction = predict_model2(new_output, m2_w0, m2_w1, m2_b1, m2_b2)
        # print(prediction)
        m2_loss = np.abs(prediction - target)
        #check if loss is within bound (~62km)
        if ((m2_loss < [0.45, 0.52]).all()): m2_passed += 1
        m2_losses.append(m2_loss)
        print(f"M2 Loss: {m2_loss}")
        

    print("# of Samples:", len(labels))
    print("# of Passed:", m2_passed)
    print(f"M1 Accuracy: {100*m1_passed/len(labels):0.3f}%")
    print(f"M2 Accuracy: {100*m2_passed/len(labels):0.3f}%")
    print(f"Average loss: {sum([(coord[0]+coord[1])/2 for coord in m2_losses])/len(labels):0.3f}")
    print(f"Lowest loss: {min([(coord[0]+coord[1])/2 for coord in m2_losses]):0.3f}")
    print(f"Highest loss: {max([(coord[0]+coord[1])/2 for coord in m2_losses]):0.3f}")

    num_samples = int(input("> ").strip())



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

    print("# of Samples:", len(labels))
    print("# of Passed:", passed)
    print(f"Accuracy: {100*passed/len(labels):0.3f}%")
    print(f"Average loss: {sum(losses)/len(labels):0.3f}")
    print(f"Lowest loss: {min(losses):0.3f}")
    print(f"Highest loss: {max(losses):0.3f}")

    num_samples = int(input("> ").strip())
'''
