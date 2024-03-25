'''
Neural Network for predicting Enemy missile strikes in Ukraine
@Date: 02/26/2024
@Author: Steven Hoodikoff
'''

import math
import numpy as np
import matplotlib.pyplot as plt


def relu(z): #Rectified Linear Unit activation function
    #change all negative values to 0 
    #e.g. [53, 13, -4] => [53, 13, 0]
    return np.maximum(0, z)

def deriv_relu(z): #ReLU derivative
    #if element is positive, set 1, else set 0
    #e.g. [53, 13, -4] => [1, 1, 0]
    return (z > 0).astype(float)

def sigmoid(z):
    ''' Sigmoid function '''
    return 1 / (1 + math.e ** (-z))

def deriv_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))


def train(data):
    ''' Train the 3x2x2 model with given data '''
    n = 0.000001 # learning rate
    # w0 = np.array([[4.34,0.22],
    #                [21.71,1.12],
    #                [17.37,0.9]]) #input layer weight matrix
    # w1 = np.array([[91,4.73],
    #                [11.17,0.58]]) #layer 1 weight matrix
    w0 = np.array([[0.353,0.033],
                   [19.858,0.78],
                   [14.528,0.57]]) #input layer weight matrix
    w1 = np.array([[-0.25,-0.163],
                   [6.459,4.222]]) #layer 1 weight matrix

    epochs = 10000
    bias_1 = 6.649 #layer 1 bias
    bias_2 = 6.699 #layer 2 bias
    loss = 0

    for epoch in range(epochs):
        for sample_num, sample in enumerate(data):
            #sample format: [dayofweek,dayofmonth,month,latitude,longitude]
            input = sample[:3] #get date data
            target = sample[3:] #get lat,long 
            
            #---Forward prop---
            #calculate z = W_T*x + b for layer 1
            weighted_sum = np.dot(w0.T, input) #multiply weights x input vector
            z1 = np.add(weighted_sum, bias_1)

            #pass layer 0 sum to activation function
            yHat_l0 = relu(z1)

            #calculate z = W_T*x + b for layer 2
            weighted_sum = np.dot(w1.T, yHat_l0)
            z2 = np.add(weighted_sum, bias_2)

            #pass layer 1 sum to activation function
            yHat_l1 = relu(z2)

            #loss = 1/2 * (yHat - target)^2
            raw_loss = np.subtract(yHat_l1, target)
            loss = 0.5 * np.power(raw_loss, 2)
            # if (raw_loss[0] < 1.8 and raw_loss[1] < 2.1):
            #     np.subtract(loss, [0.5*loss[0], 0.5*loss[1]])

            #---Backpropagation---
            # dE/dPredict
            loss_deriv = np.subtract(yHat_l1, target) #2x1 vector
            # dPredict/dReLU for layer 1 and 2
            relu_deriv_l2 = deriv_relu(yHat_l1) #2x1 vector 
            relu_deriv_l1 = deriv_relu(yHat_l0) #2x1 vector

            #calculate dL/dWi = dL*dReLU * Wi   (i=layer)
            #first take Hadamard product of Loss' & ReLU' 
            loss_relu_dot_prod2 = np.multiply(loss_deriv, relu_deriv_l2) #2x1 vector
            loss_relu_dot_prod1 = np.multiply(loss_deriv, relu_deriv_l1) #2x1 vector

            #use dot product to adjust bias   dC/dB = dC/dZ
            bias_2 = bias_2 - n*np.dot(loss_deriv, relu_deriv_l2) #scalar
            bias_1 = bias_1 - n*np.dot(loss_deriv, relu_deriv_l1) #scalar

            # calculate gradient of L W.R.T weights in input layer
            #dL/dw0 = d/dw0(input*w0) = input
            partial_L_w0 = np.outer(loss_relu_dot_prod1, input)

            # calculate gradient of L W.R.T weights in layer 1
            #dL/dw1 = d/dw1(yHat_l0*w1) = yHat_l0
            partial_L_w1 = np.outer(loss_relu_dot_prod2, yHat_l0)
            
            # update weight matricies for layer 0 and 1
            # w = w - a * dE/dw
            w0 = np.subtract(w0, np.array(np.dot(n, partial_L_w0)).T)
            w1 = np.subtract(w1, np.array(np.dot(n, partial_L_w1)).T)
            
            #print("Activations")
            #print(yHat_l0)
            #print(yHat_l1)
            #print("Weights")
            #print(w0)
            #print(w1)
            #print("dE/dW")
            # if (epoch >= 791 and sample_num % 100 == 0): 
            #     if (epoch == 792): exit()
            #     print(partial_L_w0)
            #     print(partial_L_w1)
            #     print(f"Bias1: {bias_1}  |   Bias2: {bias_2}")
            #     print(f"Loss: {loss}")
        print(f"Epoch {epoch+1} loss: {loss}")

    return w0, w1, bias_1, bias_2

def read_data():
    input_file = "output.txt"

    #file format is date,latitude,longitude
    with open(input_file, "r") as file:
        data = file.readlines()[1:]
    
    #split data 70-30 to training and test data
    ''' can't split like this because it doesn't capture an entire year
    split_index = round(len(data)*0.7)
    training = data[:split_index] # first 70%
    test = data[split_index:] # last 30%
    '''
    training = []
    test = []
    #every 3rd sample goes to testing
    for i, sample in enumerate(data):
        if (i%3==0): test.append(sample)
        else: training.append(sample)
    
    # convert all data to floats
    for i, sample in enumerate(training):
        converted_sample = list(map(float, sample.split(",")))
        training[i] = converted_sample
    for i, sample in enumerate(test):
        converted_sample = list(map(float, sample.split(",")))
        test[i] = converted_sample


    return training, test


def test(w0, w1, bias_1, bias_2, data):
    ''' Test model with unseen data '''
    passed = 0
    total = len(data)
    lowest_loss = np.array([100000,100000])
    # closest_coords = np.array([0,0])
    correct_coords = {}
    losses = []

    for sample_num, sample in enumerate(data):
        input = sample[:3] #get date data
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
        #correct approx if loss is within 1.8 deg latitude & 2.1 deg longitude (~228km^2)
        if (loss[0] < 1.8 and loss[1] < 2.1): 
            passed += 1
            correct_coords[",".join(map(str,input))] = yHat_l1

        losses.append(loss)
        # if (loss[0] < lowest_loss[0] and loss[1] < lowest_loss[1]): 
        #     closest_coords = [yHat_l1, target]

    print(f"\nTotal permissible: {passed}")
    print(f"Accuracy: {passed/total*100:0.3f}%\n")
    # print(f"Coordinates: {correct_coords}")

    plt.plot(losses)
    # Calculate averages of the two values per vector
    averages = [(x + y) / 2 for x, y in losses]
    # Plot original data
    plt.plot(range(1, len(averages) + 1), averages)
    iterations = np.arange(1, len(averages) + 1)
    coefficients = np.polyfit(iterations, averages, 1)
    poly = np.poly1d(coefficients)
    plt.plot(iterations, poly(iterations), color="red", label="Regression")
    plt.ylabel("Error (deg)")
    plt.xlabel("Iteration Number")
    plt.yticks(np.arange(0, 20, step=2), minor=True)
    plt.title("Iteration Number vs Error")
    plt.show()




if __name__ == "__main__":
    training_data, test_data = read_data()
    w0, w1, b1, b2 = train(training_data)
    print(f"Weights: \n\n{w0}\n\n{w1}")
    print(f"Bias1: {b1}  |  Bias2: {b2}")
    test(w0, w1, b1, b2, test_data)


