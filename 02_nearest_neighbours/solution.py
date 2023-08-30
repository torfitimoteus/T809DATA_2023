# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

import numpy as np
import matplotlib.pyplot as plt

from tools import load_iris, split_train_test, plot_points
import help

d, t, classes = load_iris()
x, points = d[0,:], d[1:, :]
x_target, point_targets = t[0], t[1:]

def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    x = (np.sqrt(np.sum((x - y)**2)))

    return x

#print(euclidian_distance(x, points[0]))
#print(euclidian_distance(x, points[50]))


def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and and many
    points
    '''
    distances = np.zeros(points.shape[0])  ##.shape gefur (rowcount, colcount)
    for i in range(points.shape[0]):
        distances[i] = euclidian_distance(x, points[i])

    return distances

#print(euclidian_distances(x, points))


def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    '''
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points
    '''
    distances = euclidian_distances(x, points)
    x = np.argsort(distances)[:k]

    return x

#print(k_nearest(x, points, 1))
#print(k_nearest(x, points, 3))


def vote(targets, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    # Count the occurrences of each class label
    class_counts = np.bincount(targets)

    # Find the class label with the highest count
    most_common_class = np.argmax(class_counts)

    return most_common_class

#print(vote(np.array([0,0,1,2]), np.array([0,1,2])))
#print(vote(np.array([1,1,1,1]), np.array([0,1])))


def knn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    nearest_indices = k_nearest(x, points, k)
    nearest_targets = point_targets[nearest_indices]
    predicted_class = vote(nearest_targets, classes)

    return predicted_class

#print(knn(x, points, point_targets, classes, 1))
#print(knn(x, points, point_targets, classes, 5))
#print(knn(x, points, point_targets, classes, 150))

(d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio=0.8)

def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    predictions = []
    for i, point in enumerate(points):
        removed_point = help.remove_one(points, i) #exclude current point from dataset
        removed_point_target = help.remove_one(point_targets, i) #exclude current point from dataset
        predictions.append(knn(point, removed_point, removed_point_target, classes, k))
        

    return np.array(predictions)

predictions = knn_predict(d_test, t_test, classes, 10)
#print(predictions)


def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    predictions = knn_predict(points, point_targets, classes, k)
    correct_predictions = np.sum(predictions == point_targets)
    accuracy = correct_predictions / len(points)

    return accuracy

#print(knn_accuracy(d_test, t_test, classes, 10))


def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    # Predict class labels for the points using knn_predict
    predictions = knn_predict(points, point_targets, classes, k)

    # Create an empty confusion matrix initialized with zeros
    confusion_matrix = np.zeros((len(classes), len(classes)))

    # Fill in the confusion matrix by comparing predicted labels with true labels
    for i in range(len(points)):
        true_label = point_targets[i]
        predicted_label = predictions[i]
        confusion_matrix[predicted_label][true_label] += 1

    return confusion_matrix

#print(knn_confusion_matrix(d_test, t_test, classes, 10))


def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
) -> int:
    optimal_k = None
    optimal_acc = 0
    for k in range(1, points.shape[0]):
        new_accuracy = knn_accuracy(points, point_targets, classes, k)
        if new_accuracy > optimal_acc:
            optimal_k = k
            optimal_acc = new_accuracy

    return optimal_k

#print(best_k(d_train, t_train, classes))


def knn_plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
):
    incorrect = 'red'
    correct = 'green'
    colors = ['yellow', 'purple', 'blue']
    predictions = knn_predict(points, point_targets, classes, k)
    c_map = [colors[int(i)] for i in predictions]
    correctness = (predictions == point_targets)
    
    edge = np.array(correctness.shape[0]*[correct])
    edge[correctness == False] = incorrect

    plt.scatter(points[:,0], points[:,1], c=c_map, edgecolors=edge, linewidth=2) 

    plt.show()

knn_plot_points(d, t, classes, 3)


def weighted_vote(
    targets: np.ndarray,
    distances: np.ndarray,
    classes: list
) -> int:
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    # Remove if you don't go for independent section
    ...


def wknn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    # Remove if you don't go for independent section
    ...


def wknn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    # Remove if you don't go for independent section
    ...


def compare_knns(
    points: np.ndarray,
    targets: np.ndarray,
    classes: list
):
    # Remove if you don't go for independent section
    ...
