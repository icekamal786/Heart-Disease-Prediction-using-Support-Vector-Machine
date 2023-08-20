import numpy as np
from numpy import linalg
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Function to split data into 5 parts and return
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    l = len(dataset)//n_folds
    for i in range(n_folds):
        dataset_split.append(dataset[i*l:(i+1)*l])
    return dataset_split

# Kernel functions
def linear(x1, x2, p=1, sigma = None):
    return np.dot(x1, x2)

def polynomial(x1, x2, p=2, sigma = None):
    return (1 + np.dot(x1, x2)) ** p

def rbf(x1, x2, p=None, sigma=1):
    return np.exp(-linalg.norm(x1-x2)**2 / (2 * (sigma ** 2)))

# Constraints for minimization
def constraint1(alpha):
    return alpha

def constraint2(alpha, Y):
    return np.sum(alpha * Y)

class SVM:

    # Initialize class variables
    def __init__(self, kernel, C, p=2, sigma = 1):
        self.C = C              # Hyperparameter
        self.kernel = kernel    # Kernel Type
        self.p = p              # Degree of kernel (if polynomial kernel is used)
        self.sigma = sigma
        self.H = None           # Hessian Matrix
        self.W = None           # Weights
        self.B = None           # Bias
        self.X = None
        self.Y = None
        self.alpha = None

    # Lagrangian Function
    def objective(self, alpha):
        return 0.5 * np.dot(alpha, np.dot(self.H, alpha) + np.sum(alpha*np.log(alpha/(self.C-alpha))) - self.C*np.sum(np.log(self.C/(self.C-alpha))) )


    def Train(self, X, Y):
        # Evaluate Hessian Matrix depending on the kernel
        self.X = X
        self.Y = Y
        H = np.zeros((X.shape[0],X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                H[i,j] = self.kernel(X[i], X[j], p=self.p, sigma = self.sigma)*Y[i]*Y[j]
        self.H = H

        
        # Define the bounds for the alpha values
        bounds = [(0, self.C) for i in range(X.shape[0])]

        # Define the constraints
        constraints = [{'type': 'ineq', 'fun': constraint1},
               {'type': 'eq', 'fun': constraint2, 'args': (Y,)}]

        # Solve the SVM using scipy.optimize.minimize
        init_alpha = np.ones(X.shape[0])*self.C*0.5
        solution = minimize(self.objective, init_alpha, method='SLSQP', bounds=bounds, constraints=constraints)

        # Extract the Lagrange multipliers from the solution
        alpha = solution.x

        # Support vectors have non zero lagrange multipliers
        n_samples = X.shape[0]
        sv = alpha > 1e-3
        ind = np.arange(len(alpha))[sv]
        self.a = alpha[sv]
        self.sv = X[sv]
        self.sv_y = Y[sv]

        # Compute the weight vector and bias term
        alpha = alpha.reshape(-1,1)
        self.alpha = alpha
        Y = Y.reshape(-1,1)

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        # Intercept
        self.B = 0
        for n in range(len(self.a)):
            self.B += self.sv_y[n]
            self.B -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.B /= len(self.a)

    
    def Predict(self, X_test, Y_test,doPrint):
        # Compute the dot product of the weight matrix and the test data, and add the bias term
        y_predict = np.zeros(len(X_test))
        for i in range(len(X_test)):
            temp = 0
            for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    temp += a * sv_y * self.kernel(X_test[i], sv)
            y_predict[i] = temp

        accuracy = self.Print_Results(Y_test, y_predict+self.B , doPrint=print)
        return accuracy

    def Print_Results(self, y_true, y_pred, doPrint):
        # Calculate the number of true positives, false positives, and false negatives
        t_p = 0
        t_n = 0
        f_p = 0
        f_n = 0

        n_test_pos = 0
        n_test_neg = 0

        for (i,j) in zip(y_true,y_pred):
            if (i == 1) and (j > 0):
                t_p=t_p+1
                n_test_pos = n_test_pos+1
            elif (i == -1) and (j <= 0):
                t_n=t_n+1
                n_test_neg = n_test_neg+1
            elif (i == -1) and (j > 0):
                f_p=f_p+1
                n_test_neg = n_test_neg+1
            elif (i == 1) and (j <= 0):
                f_n=f_n+1
                n_test_pos = n_test_pos+1

        # Calculate accuracy, recall, and F1 score
        accuracy = (t_p+t_n)/(t_p+t_n+f_p+f_n)

        precision_pos = t_p/(f_p+t_p)
        precision_neg = t_n/(f_n+t_n)

        recall_pos = t_p/(f_n+t_p)
        recall_neg = t_n/(f_p+t_n)

        f1_score_pos = (2*precision_pos*recall_pos)/(precision_pos+recall_pos)
        f1_score_neg = (2*precision_neg*recall_neg)/(precision_neg+recall_neg)

        if print:
            # Print results
            print("\t\tprecision\trecall\t\tf1-score\tsupport")
            print()
            print("neg (-1)\t{neg_pre:.2f}\t\t{neg_recall:.2f}\t\t{neg_f1_score:.2f}\t\t{neg_sup}".format(neg_pre=precision_neg,neg_recall= recall_neg,neg_f1_score=f1_score_neg,neg_sup=n_test_neg))
            print("pos (1)\t\t{pos_pre:.2f}\t\t{pos_recall:.2f}\t\t{pos_f1_score:.2f}\t\t{pos_sup}".format(pos_pre=precision_pos,pos_recall= recall_pos,pos_f1_score=f1_score_pos,pos_sup=n_test_pos))
            print()
            print("accuracy\t\t\t\t\t{acc:.2f}\t\t{acc_sup}".format(acc=accuracy,acc_sup=n_test_pos+n_test_neg))
            print()
        return accuracy


def main():
    # Read Dataset
    dataset = pd.read_csv("./heart.csv")

    dataset['target'] = dataset['target'].replace(0, -1)
    shuffled_data= dataset.sample(frac=1)
    train_data_num = int(shuffled_data.shape[0]*0.8)
    train_data = shuffled_data.iloc[:train_data_num]
    test_data = shuffled_data.iloc[train_data_num:]

    # Normalize train data
    numerical_features = ['age', 'cp', 'trestbps', 'chol', 'restecg', 'thalach', 'oldpeak', 'slope', 'ca', 'thal']

    # compute mean and standard deviation of each feature
    means = train_data[numerical_features].mean()
    stds = train_data[numerical_features].std()

    # apply normalization formula to each value in each feature
    train_data[numerical_features] = (train_data[numerical_features] - means) / stds

    # normalize test data
    test_data[numerical_features] = (test_data[numerical_features] - means) / stds


    # Spilt in X and Y
    X_train = train_data.drop('target', axis=1)
    Y_train = train_data['target']
    X_test = test_data.drop('target', axis=1)
    Y_test = test_data['target']

    best_C = 100
    best_P = 2
    best_S = 2.5

########################### 5-fold cross validation for hyperparameter tuning ##########################
#                                     (Might take a lot of time)

########### Comment this part of code to skip hyperparameter tuning and used pre-tuned values ############


    folds = cross_validation_split(train_data, 5)

    # Hyperparameter tuning for C
    C_List = [0.1,1,10,100]
    best_C = -1
    best_accu = -1
    
    print("Hyperparameter tuning for C ....")
    for c in C_List:
        accuracy = 0
        print("Testing for C = " + str(c))
        for i in range(5):
            print("Fold "+str(i+1)+"/5")
            train_set = pd.concat(folds[j] for j in range(5) if j!=i)
            X_train_set = train_set.drop('target', axis=1)
            Y_train_set = train_set['target']
            X_validation_set = folds[i].drop('target', axis=1)
            Y_validation_set = folds[i]['target']
            myLinearSVM = SVM(kernel=linear, C=c)
            myLinearSVM.Train(X_train_set.to_numpy(),Y_train_set.to_numpy())
            accuracy += myLinearSVM.Predict(X_validation_set.to_numpy(), Y_validation_set.to_numpy(),False)
        accuracy /= 5

        if accuracy > best_accu:
            best_accu = accuracy
            best_C = c
    
    print("Best value of C: "+ str(best_C))


    # Hyperparameter tuning for p
    P_List = [2,3,4,5]
    best_P = -1
    best_accu = -1
    print("Hyperparameter tuning for P ....")
    for p in P_List:
        accuracy = 0
        print("Testing for P = " + str(p))
        for i in range(5):
            print("Fold "+str(i+1)+"/5")
            train_set = pd.concat(folds[j] for j in range(5) if j!=i)
            X_train_set = train_set.drop('target', axis=1)
            Y_train_set = train_set['target']
            X_validation_set = folds[i].drop('target', axis=1)
            Y_validation_set = folds[i]['target']
            myPolynomialSVM = SVM(kernel=polynomial, C=100,  p=p)
            myPolynomialSVM.Train(X_train_set.to_numpy(),Y_train_set.to_numpy())
            accuracy += myPolynomialSVM.Predict(X_validation_set.to_numpy(), Y_validation_set.to_numpy(),False)
        accuracy /= 5

        if accuracy > best_accu:
            best_accu = accuracy
            best_P = p
    
    print("Best value of P: "+ str(best_P))

    # Hyperparameter tuning for sigma
    S_List = [0.1,0.5,2.5,12.5,62.5]
    best_S = -1
    best_accu = -1
    print("Hyperparameter tuning for Sigma....")
    for s in S_List:
        accuracy = 0
        print("Testing for s = " + str(s))
        for i in range(5):
            print("Fold "+str(i+1)+"/5")
            train_set = pd.concat(folds[j] for j in range(5) if j!=i)
            X_train_set = train_set.drop('target', axis=1)
            Y_train_set = train_set['target']
            X_validation_set = folds[i].drop('target', axis=1)
            Y_validation_set = folds[i]['target']
            myRBF_SVM = SVM(kernel=rbf, C=100, sigma=s)
            myRBF_SVM.Train(X_train_set.to_numpy(),Y_train_set.to_numpy())
            accuracy += myRBF_SVM.Predict(X_validation_set.to_numpy(), Y_validation_set.to_numpy(),False)
        accuracy /= 5

        if accuracy > best_accu:
            best_accu = accuracy
            best_S = s
        
    print("Best value of P: "+ str(best_P))

##################################### Hyperparameters tuning ends ####################################            


############################ Training and Testing on best hyperparameters ############################

    # Linear Kernel
    print("Training Linear Kernel SVM...")
    myLinearSVM = SVM(kernel=linear, C=best_C)
    myLinearSVM.Train(X_train.to_numpy(),Y_train.to_numpy())
    myLinearSVM.Predict(X_test.to_numpy(), Y_test.to_numpy(),doPrint=True)

    # Polynomial Kernel
    print("Training Polynomial Kernel SVM...")
    myPolynomialSVM = SVM(kernel=polynomial, C=best_C, p=best_P)
    myPolynomialSVM.Train(X_train.to_numpy(),Y_train.to_numpy())
    myPolynomialSVM.Predict(X_test.to_numpy(), Y_test.to_numpy(),doPrint=True)

    # Radial Basis Function (RBF) Kernel
    print("Training RBF Kernel SVM...")
    myRBF_SVM = SVM(kernel=rbf, C=best_C, sigma= best_S)
    myRBF_SVM.Train(X_train.to_numpy(),Y_train.to_numpy())
    myRBF_SVM.Predict(X_test.to_numpy(), Y_test.to_numpy(),doPrint=True)

    # SVM Using Library function
    print("SVM using Sklearn...")
    svclassifier = LinearSVC()
    svclassifier.fit(X_train, Y_train)
    Y_pred = svclassifier.predict(X_test)
    
    print(classification_report(Y_test,Y_pred))


if __name__ == "__main__":
    main()
