'''
Neural Network for predicting Enemy missile strikes in Ukraine
@Date: 02/26/2024
@Author: Steven Hoodikoff
'''

import numpy as np
import matplotlib.pyplot as plt
from util import Util

class StrikeNN:
    def __init__(self, data=[]):
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


    def train(self, epochs=1000):
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
                


    def test(self):
        ''' Test model with unseen data '''
        passed = 0
        total = len(self.data)
        # closest_coords = np.array([0,0])
        correct_coords = {}
        losses = []

        for sample_num, sample in enumerate(self.data):
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


    def predict(self, date):
        ''' Receive a date and create a prediction '''
        util = Util()
        df = util.get_formatted_date(date) #convert date to accepted input type (DoW, DoM, month)
        date_input = [df["day_of_week"], df["day_of_month"], df["month"]]
        print("input:",date_input)
        #input layer
        weighted_sum = np.dot(self.w0.T, date_input)
        z1 = weighted_sum + self.bias_1
        yHat_l0 = self.relu(z1)
        #layer 1
        weighted_sum = np.dot(self.w1.T, yHat_l0)
        z2 = weighted_sum + self.bias_2
        yHat_l1 = self.relu(z2) #final answer
        print("prediction:",yHat_l1)
        return yHat_l1