'''
2 (Regional & Precision) Neural Networks for predicting Enemy missile strikes in Ukraine

How to:
- Get data from util and separate into training and testing batches
- Instantiate StrikeNN object
- Train model 1 (regional) using train_model1() method
- Test model 1 using test_model1() method
- Train model 2 (precision) using train_model2() method
- Test model 2 using test_model2() method
- Make a prediction using predict() method  

Example:
network = StrikeNN(data=util.training_data, test_data=util.testing_data)
network.train_model1(epochs=100) 
network.train_model2(epochs=100)
network.test_model1()
network.test_model2()
date = [4,26,2]
network.predict(date)


@Date: 02/26/2024
@Author: Steven Hoodikoff
'''

import numpy as np
import matplotlib.pyplot as plt
from util import Util

class StrikeNN:
    def __init__(self, data=[], test_data=[]):
        self.n = 0.0000001 # learning rate
        self.w0 = np.array([[0.353,0.033],
                            [19.858,0.78],
                            [14.528,0.57]]) #input layer weight matrix
        self.w1 = np.array([[-0.25,-0.163],
                            [6.459,4.222]]) #layer 1 weight matrix

        self.bias_1 = 6.649 #layer 1 bias
        self.bias_2 = 6.699 #layer 2 bias
        self.loss = 0
        self.data = data
        self.test_data = test_data
        self.samples = len(data)
        self.accuracy = 0
        self.prediction_confidence = 0
        
    def relu(self, z): #Rectified Linear Unit activation function
        #change all negative values to 0 
        #e.g. [53, 13, -4] => [53, 13, 0]
        return np.maximum(0, z)

    def deriv_relu(self, z): #ReLU derivative
        #if element is positive, set 1, else set 0
        #e.g. [53, 13, -4] => [1, 1, 0]
        return (z > 0).astype(float)


    def train_model1(self, epochs=1000):
        ''' Train the 3x2x2 model with given data '''

        for epoch in range(epochs):
            epoch_loss = 0
            for sample_num, sample in enumerate(self.data):
                #sample format: [dayofweek,dayofmonth,month,latitude,longitude]
                input = sample[:3] #get date data
                target = sample[3:] #get lat,long 
                
                #---Forward prop---
                #calculate z = W_T*x + b for layer 1
                weighted_sum = np.dot(self.w0.T, input) #multiply weights x input vector
                z1 = weighted_sum + self.bias_1

                #pass layer 0 sum to activation function
                yHat_l0 = self.relu(z1)

                #calculate z = W_T*x + b for layer 2
                weighted_sum = np.dot(self.w1.T, yHat_l0)
                z2 = weighted_sum + self.bias_2

                #pass layer 1 sum to activation function
                yHat_l1 = self.relu(z2)

                #loss = 1/2 * (yHat - target)^2
                self.loss = 0.5 * np.power(yHat_l1 - target, 2)
                epoch_loss += self.loss

                #---Backpropagation---
                # dE/dPredict
                loss_deriv = np.subtract(yHat_l1, target) #2x1 vector
                # dPredict/dReLU for layer 1 and 2
                relu_deriv_l2 = self.deriv_relu(z2) #2x1 vector 
                relu_deriv_l1 = self.deriv_relu(z1) #2x1 vector

                #calculate dL/dWi = dL*dReLU * Wi   (i=layer)
                #first take Hadamard product of Loss' & ReLU' 
                loss_relu_dot_prod2 = np.multiply(loss_deriv, relu_deriv_l2) #2x1 vector
                loss_relu_dot_prod1 = np.multiply(loss_deriv, relu_deriv_l1) #2x1 vector

                #use dot product to adjust bias   dC/dB = dC/dZ
                self.bias_2 = self.bias_2 - self.n*np.dot(loss_deriv, relu_deriv_l2) #scalar
                self.bias_1 = self.bias_1 - self.n*np.dot(loss_deriv, relu_deriv_l1) #scalar

                # calculate gradient of L W.R.T weights in input layer
                #dL/dw0 = d/dw0(input*w0) = input
                partial_L_w0 = np.outer(loss_relu_dot_prod1, input)

                # calculate gradient of L W.R.T weights in layer 1
                #dL/dw1 = d/dw1(yHat_l0*w1) = yHat_l0
                partial_L_w1 = np.outer(loss_relu_dot_prod2, yHat_l0)
                
                # update weight matricies for layer 0 and 1
                # w = w - a * dE/dw
                self.w0 = self.w0 - (self.n * partial_L_w0).T
                self.w1 = self.w1 - (self.n * partial_L_w1).T
                


    def test_model1(self):
        ''' Test model with unseen data '''
        passed = 0
        total = len(self.test_data)
        correct_coords = {}
        losses = []

        for sample_num, sample in enumerate(self.test_data):
            input = sample[:3] #get date data
            target = sample[3:] #get lat,long 

            weighted_sum = np.dot(self.w0.T, input) #multiply weights x input vector
            z1 = np.add(weighted_sum, self.bias_1)
            #pass layer 0 sum to activation function
            yHat_l0 = self.relu(z1)

            #calculate z = W_T*x + b for layer 2
            weighted_sum = np.dot(self.w1.T, yHat_l0)
            z2 = np.add(weighted_sum, self.bias_2)

            #pass layer 1 sum to activation function
            yHat_l1 = self.relu(z2)

            self.loss = np.absolute(np.subtract(yHat_l1, target))
            #correct approx if loss is within 1.8 deg latitude & 2.1 deg longitude (~228km^2)
            if (self.loss[0] < 1.8 and self.loss[1] < 2.1): 
                passed += 1
                correct_coords[",".join(map(str,input))] = yHat_l1

            losses.append(self.loss)

        print(f"\nTotal permissible: {passed}")
        print(f"Accuracy: {passed/total*100:0.3f}%\n")


    def predict_m1(self, date, formatted=False):
        ''' Receive a date and create a prediction '''
        date_input = date
        if (not formatted):
            util = Util()
            df = util.get_formatted_date(date) #convert date to accepted input type (DoW, DoM, month)
            date_input = [df["day_of_week"], df["day_of_month"], df["month"]]
        # print("input:",date_input)

        #---MODEL 1 CALCULATIONS (REGIONAL)---
        #input layer
        weighted_sum = np.dot(self.w0.T, date_input)
        z1 = weighted_sum + self.bias_1
        yHat_l0 = self.relu(z1)
        #layer 1
        weighted_sum = np.dot(self.w1.T, yHat_l0)
        z2 = weighted_sum + self.bias_2
        yHat_l1 = self.relu(z2) #final answer from model 1

        # print("M1 prediction:",yHat_l1)
        return yHat_l1
    
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


    def get_samples(self, coord_range):
        ''' Get all samples within a coordinate range
        :param: data - list of all samples
        :param: coord_range - list of lower & upper bounds for a coordinate
        :return: new_data - list of all samples within range of input coordinates 
        '''
        new_training_data = []
        new_testing_data = []
        counter = 0
        for sample in self.data:
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

    def train_model2(self, data):
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

        epochs = 200
        bias_1 = 3.656516218491046 #layer 1 bias (67.742%)
        bias_2 = 2.636866698491047 #layer 2 bias (67.742%)
        loss = 0
        samples = len(data)

        for epoch in range(epochs):
            epoch_loss = 0
            break
            for sample_num, sample in enumerate(data):
                prediction = predict_model1(sample[:3], self.w0, self.w1, self.bias_1, self.bias_2) #get prediction from model1
                input = convert_model1_output(prediction)#get coordinate range
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

                #loss = 1/2 * (yHat - target)^2
                loss = 0.5 * np.power(yHat_l1 - target, 2)
                epoch_loss += loss
                # if (raw_loss[0] < 1.8 and raw_loss[1] < 2.1):
                #     np.subtract(loss, [0.5*loss[0], 0.5*loss[1]])

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
                # print("lrdp1:",loss_relu_dot_prod1.shape)
                # print("lrdp2:",loss_relu_dot_prod2.shape)
                # print("rd1:", relu_deriv_l1.shape)
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
                # print("pLw0:", partial_L_w0.shape)
                # print("pLw1:", partial_L_w1.shape)
                # print("b1:", bias_1)
                # print("b2:", bias_2)
                # update weight matricies for layer 0 and 1
                # w = w - a * dE/dw
                w0 = w0 - (n * partial_L_w0).T
                w1 = w1 - (n * partial_L_w1).T
                
            #print(f"Epoch {epoch+1} loss: {epoch_loss/samples}")

        return w0, w1, bias_1, bias_2 


    def test_model2(self, w0, w1, bias_1, bias_2, m1_w0, m1_w1, m1_b1, m1_b2, data):
        ''' Test 4x3x2 model (precision model) with unseen data '''
        passed = 0
        total = len(data)
        correct_coords = {}
        losses = []

        for sample_num, sample in enumerate(data):
            prediction = self.predict_m1(sample[:3]) #get prediction from model1
            input = self.convert_model1_output(prediction)#get coordinate range
            target = sample[3:] #get lat,long 

            weighted_sum = np.dot(w0.T, input) #multiply weights x input vector
            z1 = np.add(weighted_sum, bias_1)
            #pass layer 0 sum to activation function
            yHat_l0 = self.relu(z1)

            #calculate z = W_T*x + b for layer 2
            weighted_sum = np.dot(w1.T, yHat_l0)
            z2 = np.add(weighted_sum, bias_2)

            #pass layer 1 sum to activation function
            yHat_l1 = self.relu(z2)

            loss = np.absolute(np.subtract(yHat_l1, target))
            #correct approx if loss is within 0.45 deg latitude & 0.525 deg longitude (~57km^2)
            if ((loss < [0.45, 0.52]).all()): 
                passed += 1
                correct_coords[",".join(map(str,input))] = yHat_l1

            losses.append(loss)

        print(f"\nM2 Total permissible: {passed}")
        print(f"M2 Accuracy: {passed/total*100:0.3f}%\n")


    def predict_model2(self, date, formatted=False):
        ''' Create a prediction using the Model 2 (precision) network 
        :return: list  [latitude, longitude]
        '''

        #Model 1 computation
        m1_prediction = self.predict_m1(date, formatted)
        #Convert output to range
        m1_pred_range = self.convert_model1_output(m1_prediction)
        m2_train_data, m2_test_data = self.get_samples(m1_pred_range)
        m2_w0, m2_w1, m2_b1, m2_b2 = self.train_model2(m2_train_data)
        #input layer
        weighted_sum = np.dot(m2_w0.T, m1_pred_range)
        z1 = weighted_sum + m2_b1
        yHat_l0 = self.relu(z1)

        #layer 1
        weighted_sum = np.dot(m2_w1.T, yHat_l0)
        z2 = weighted_sum + m2_b2
        yHat_l1 = self.relu(z2) #final answer

        # print(f"M2 prediction:", yHat_l1)
        return yHat_l1
    
    def predict(self, date, formatted=False):
        ''' User method for creating a coordinate prediction 
            given a date.
        :param: date (int list) - [day_of_week, day_of_month, month_of_year]
        :return: float list - [latitude, longitude]
        '''
        return self.predict_model2(date, formatted)