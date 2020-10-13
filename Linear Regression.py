from sklearn.linear_model import LinearRegression 

import util
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta
        self.epochs = 100 #this is the amount of times the program will run

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        for i in range(self.epochs): #starting at zero, iterate until the number of epochs
            #loss = 0 #to get sum of loss function, you have to start at 0
            normal = 0 #to get sum of the normal equation, you have to start at 0

                #loss = (((X[i]*self.theta) - y[i]).transpose()) * ((X[i]*self.theta) - y[i])
            normal = np.sum((((X.transpose())*X)**-1)*((X.transpose())*y)) #this is the normal equation
            self.theta = self.theta - 0.000002*(normal) #use the normal equation and alpha to update theta
  
        
            
        # *** END CODE HERE ***

    def fit_GD(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the gradient descent algorithm.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        for i in range(self.epochs): #starting at zero, iterate until the number of epochs
            #loss = 0 #to get sum of loss function, you have to start at 0
            loss_derivative = 0 #to get sum of the normal equation, you have to start at 0
            for i in range(len(y)): #start from 0, iterate until the end of the dataset outcomes
                #loss += ((y[i]-(self.theta*X[i]))**2)/2 
                loss_derivative += (y[i] - self.theta*X[i])*(-X[[i]]) #this is the equation of the derivative of the loss function
                #print(X[i])
            self.theta = self.theta - 0.000002*loss_derivative #use the loss function derivative to update theta
         
        # *** END CODE HERE ***

    def fit_SGD(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the stochastic gradient descent algorithm.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        for i in range(self.epochs):#starting at zero, iterate until the number of epochs
            #loss = 0
            for i in range(len(y)): #this time a nesed loop is required because we are not updating theta with the sum of the derivative anymore, we are updating theta with eath derivative
                for j in range(len(X)):
                    self.theta = self.theta - 0.000002*(self.theta*X[i] - y[i])*X[i] #theta is being updated

                
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        for i in range(self.epochs):
            poly = 0 
            
            for i in range(len(X)): 
                for j in range (k):
                    poly += (self.theta*(X[i]**j))
                    #print(poly)
            self.theta = self.theta - 0.0000002*poly
            #print(self.theta)
            #print(X)
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        for i in range(self.epochs):
            sin = 0 
            for i in range(len(X)): 
                for j in range (k):
                    sin += self.theta*(X[i]**j)*math.sin(X[i])
                   
            self.theta = self.theta - 0.0000002*sin
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        
        y_hat = [] #create an empty list which will store each predicted y value
        for i in range (len(X)): #start from 0, iterate until the end of the input
            y_hat.append(self.theta*X[i]) #append all of the predicted y values to the empty list
        return np.array(y_hat) #return the list filled with predicted y values
        # *** END CODE HERE ***


def run_exp(data, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x= np.array(data["x"].values) #this variable stores the input of the dataset
    train_y= np.array(data["y"].values) #this variable stores the output of the dataset
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    '''
    Our objective is to train models and perform predictions on plot_x data
    '''
    # *** START CODE HERE ***
    GDplot = LinearModel(0.0000001) #create an object for GD
    SGDplot = LinearModel(0.000001) #create an object for SGD
    Normalplot = LinearModel(0.000001) #create an object for normal equations
    polyplot = LinearModel(0.000001) #create an object for polynomial
    polyplot2 = LinearModel(0.000001) #create an object for polynomial
    polyplot3 = LinearModel(0.000001) #create an object for polynomial
    sinplot =  LinearModel(0.000001) #create an object for sin
    
    polyplot.create_poly(3,train_x) #run the training for the poly object with the appropiate function
    polyplot2.create_poly(5,train_x) #run the training for the poly object with the appropiate function
    polyplot3.create_poly(10,train_x)
    sinplot.create_sin(10,train_x) #run the training for the sin object with the appropiate function
    lr= LinearRegression().fit(train_x.reshape(-1, 1), train_y)
    SKLEARNpredict= lr.predict(train_x.reshape(-1, 1))
    Normalplot.fit(train_x,train_y) #run the training for the normal object with the appropiate function
    Normalpredict = Normalplot.predict(train_x) #predict future y values of the normal equation object
    GDplot.fit_GD(train_x,train_y) #run the training for the GD object with the appropiate function
    SGDplot.fit_SGD(train_x,train_y) #run the training for the SGD object with the appropiate function
    
    GDpredict = GDplot.predict(train_x) #predict future y values of the GD object
    SGDpredict = SGDplot.predict(train_x) #predict future y values of the SGD object
    Polypredict = polyplot.predict(train_x)
    Polypredict2 = polyplot2.predict(train_x)
    Polypredict3 = polyplot3.predict(train_x)
    Sinpredict = sinplot.predict(train_x)
    #print(train_y)
    # *** END CODE HERE ***
    '''
    Here plot_y are the predictions of the linear model on the plot_x data
    '''
    plots= [(GDpredict,'Gradient Descent'), (SGDpredict,'Stochastic Gradient Descent'), (Normalpredict,'Normal Equations'), (SKLEARNpredict,'SkLearn')]
    #plt.ylim(-2, 2)
    
        
    #plt.scatter(train_x, GDpredict, label = 'Gradient Descent') #plot the scatter plot for GD using the trained input as x and the prediction of output as y
    #plt.scatter(train_x, SGDpredict, color = 'red', label = 'Stochastic Descent') #plot the scatter plot for SGD using the trained input as x and the prediction of output as y
    #plt.scatter(train_x, Normalpredict, color = 'green', label = 'Normal Equations') #plot the scatter plot for normal equations using the trained input as x and the prediction of output as y
    plt.scatter(train_x, Polypredict,label = 'k = 3', color = 'red')
    plt.scatter(train_x, Polypredict2,label = 'k = 5', color = 'blue')
    plt.scatter(train_x, Polypredict3,label = 'k = 10', color = 'green')
    #plt.scatter(train_x, Sinpredict,label = 'sin')
    plt.legend()
    #plt.savefig(filename)
    #plt.clf()
def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***
    
    train = pd.read_csv(train_path)
    #print(train)
    run_exp(train)
    # *** END CODE HERE ***


if __name__ == '__main__':
    main(train_path='trainq1.csv',
        small_path='smallq1.csv',
        eval_path='testq1.csv')
