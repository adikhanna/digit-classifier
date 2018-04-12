import sys
import decimal
import math
import time
import operator
import itertools
import prettytable
import numpy as np
import scipy.spatial.distance as d
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def nearestNeighbor(train_digits, test_digits, k):

    print '\n'

    start = time.time()

    train_classes = []
    test_classes = []

    train_pixels = []
    test_pixels = []

    for x in train_digits:
        train_classes.append(x[0])
        train_pixels.append(x[1])

    for y in test_digits:
        test_classes.append(y[0])
        test_pixels.append(y[1])

    train_pixels = np.array(train_pixels)
    test_pixels = np.array(test_pixels)

    testData_length = len(test_classes)
    trainData_length = len(train_classes)

    solution = []

    for i in range(testData_length):
        neighbors = {}
        test = test_pixels[i].flatten()

        for j in range(trainData_length):
            train = train_pixels[j].flatten()
            neighbors[j] = d.hamming(test, train)

        neighbors = sorted(neighbors.items(), key=operator.itemgetter(1))
        nn = list(neighbors[:k])

        prediction = getPrediction(train_classes, nn)
        solution.append(prediction)

    end = time.time()

    elapsed_time = end - start
    print 'Running time is: ' + str(elapsed_time) + 's'
    print '\n'

    # Overall Test Set Accuracy

    setLen = len(solution)
    acc = 0

    for i in range(setLen):

        a = int(solution[i])
        b = int(test_classes[i])

        if a == b:
            acc = acc + 1
        else:
            continue

    accuracy = (acc / float(setLen))*(100.0)

    print 'Overall Test Set Accuracy: ' + str(accuracy) + '%'

    print '\n'

    # Confusion Matrix

    # conMat = []
    # dim = 10
    #
    # conMat = np.array(conMat)
    # p = prettytable.PrettyTable()
    #
    # for row in conMat:
    #     p.add_row(row)
    #
    # print p.get_string(header=False, border=False)


def getPrediction(train, neighbors):
    votes = {}
    nLength = len(neighbors)
    i = 0

    while (i < nLength):

        index = neighbors[i][0]
        item = train[index]

        if item in votes:
            votes[item] = votes[item] + 1
        else:
            votes[item] = 1

        i = i + 1

    result = max(votes, key=votes.get)
    return result

def perceptrons(train_digits, test_digits, data_set):

    eta = 0.25 # learning rate decay
    epochs = 5

    confusion_matrix = [[0 for x in range(10)] for y in range(10)]

    if data_set == '0':
        digit_set = train_digits
    else:
        digit_set = test_digits

    X_vector = [[0 for x in range(1024)] for y in range(len(digit_set))]

    W_vector = [[1 for x in range(1024)] for y in range(10)]

    count = -1

    for digit in digit_set:

        count = count + 1

        for i in range(32):
            for j in range(32):

                X_vector[count][(i * 32) + j] = digit[1][i][j]


    predicted_class = [0 for x in range(len(digit_set))]

    for y in range(epochs):
        print y
        count = -1

        for digit in digit_set:

            count = count + 1

            max_value = 0
            max_class = 0

            for class_num in range(10):

                value = 0

                for x in range(1024):
                    value = value + (int(X_vector[count][x]) * W_vector[class_num][x])

                if value > max_value:
                    max_value = value
                    max_class = class_num

            predicted_class[count] = max_class

            if (y == 4):
                confusion_matrix[int(digit[0])][max_class] = confusion_matrix[int(digit[0])][max_class] + 1

            if max_class == digit[0]:
                continue
            else:
                for x in range(1024):
                    W_vector[int(digit[0])][x] = float(W_vector[int(digit[0])][x]) +(eta * float(X_vector[count][x]))
                    W_vector[max_class][x] = float(W_vector[max_class][x]) - (eta * float(X_vector[count][x]))

    class_count = [0 for x in range(10)]
    for i in range(len(digit_set)):
        class_count[int(digit_set[i][0])] =  class_count[int(digit_set[i][0])] + 1

    for i in range(10):
        for j in range(10):

            confusion_matrix[i][j] = confusion_matrix[i][j] / float(class_count[i])

        print confusion_matrix[i],"\n"

    right = 0
    wrong = 0

    for y in range(len(digit_set)):

        if (str(digit_set[y][0]) == str(predicted_class[y])):
            right = right + 1
        else:
            wrong = wrong + 1

        #print digit_set[y][0], "vs", predicted_class[y]

    if data_set == '0':
        print "training set accuracy: ", right / float(right + wrong)
    else:
        print "testing set accuracy: ", right / float(right + wrong)

    # Building Heat Maps for 2.3(a)

    for x in range(10):
        pixel_data = W_vector[x]
        pixel_2D = np.reshape(pixel_data, (32, 32))
        plt.imshow(pixel_2D, cmap='jet', interpolation='nearest')
        name = 'Class ' + str(x)
        plt.savefig(name)



def extractPixels(data_set):

    fd = open("optdigits-orig_train.txt", "r")

    train_digits = []

    for i in range(2436):

        digit_data = []

        for j in range(33):

            line = fd.readline()
            line = tuple(line)

            if (j != 32):
                digit_data.append(line)

            else:
                class_num = line[1]

        train_digits.append((class_num, digit_data))


    fd = open("optdigits-orig_test.txt", "r")

    test_digits = []

    for i in range(444):

        digit_data = []

        for j in range(33):

            line = fd.readline()
            line = tuple(line)

            if (j != 32):
                digit_data.append(line)

            else:
                class_num = line[1]

        test_digits.append((class_num, digit_data))

    # perceptrons(train_digits, test_digits, data_set)

    # run nn for all k

    for k in range(1, 26):
        print "\n"
        print "Running Nearest Neighbor Classifier with k = " + str(k)
        nearestNeighbor(train_digits, test_digits, k)
        print "\n"


extractPixels(0)


# references:
# https://towardsdatascience.com/mnist-with-k-nearest-neighbors-8f6e7003fab7
# https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
# https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
# https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
# https://stackoverflow.com/questions/29216889/slicing-a-dictionary
# https://stackoverflow.com/questions/26871866/print-highest-value-in-dict-with-key
# https://jeremykun.com/2012/08/26/k-nearest-neighbors-and-handwritten-digit-classification/
# https://stackoverflow.com/questions/13214809/pretty-print-2d-python-list
