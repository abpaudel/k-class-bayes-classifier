import numpy as np
import sys

X_train = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2])
X_test = np.genfromtxt(sys.argv[3], delimiter=",")


def pluginClassifier(X_train, y_train, X_test):
    pass


final_outputs = pluginClassifier(X_train, y_train, X_test)

np.savetxt("probs_test.csv", final_outputs, delimiter=",")