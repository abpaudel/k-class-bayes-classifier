import numpy as np
import sys

def plugin_classifier(X_train, y_train, X_test):
    n_class = len(np.unique(y_train))
    n = X_train.shape[0]
    prior = {}
    mean = {}
    cov_inv = {}
    cov_ird = {}
    posterior = []
    for label in range(n_class):
        X = X_train[y_train==label]
        prior[label] = len(X)/float(n)
        mean[label] = np.mean(X, axis=0, dtype=np.float64)
        cov = np.cov(X.T)
        cov_inv[label] = np.linalg.inv(cov)
        cov_ird[label] = 1 / np.sqrt(np.linalg.det(cov))
    for x in X_test:
        for c in range(n_class):
            diff = x - mean[c]
            posterior.append(prior[c]*cov_ird[c]*np.exp(-0.5*np.dot(diff, np.dot(cov_inv[c], diff.T))))
    pred_y = np.reshape(posterior, (len(X_test), n_class))
    s = np.reshape(np.sum(pred_y, axis=1), (-1, 1))
    pred_y = pred_y/s
    return pred_y

def main():
    X_train = np.genfromtxt(sys.argv[1], delimiter=",")
    y_train = np.genfromtxt(sys.argv[2])
    X_test = np.genfromtxt(sys.argv[3], delimiter=",")
    final_outputs = plugin_classifier(X_train, y_train, X_test)
    np.savetxt("probs_test.csv", final_outputs, delimiter=",")

if __name__ == '__main__':
    main()