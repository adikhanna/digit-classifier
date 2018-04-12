import sys
import decimal
import math
import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def digit_sort(digits):

    zero = []
    one = []
    two = []
    three = []
    four = []
    five = []
    six = []
    seven = []
    eight = []
    nine = []

    sorted_digits = []

    for i in range(len(digits)):

        if digits[i][0] == '0':
            zero.append(digits[i][1])
        elif digits[i][0] == '1':
            one.append(digits[i][1])
        elif digits[i][0] == '2':
            two.append(digits[i][1])
        elif digits[i][0] == '3':
            three.append(digits[i][1])
        elif digits[i][0] == '4':
            four.append(digits[i][1])
        elif digits[i][0] == '5':
            five.append(digits[i][1])
        elif digits[i][0] == '6':
            six.append(digits[i][1])
        elif digits[i][0] == '7':
            seven.append(digits[i][1])
        elif digits[i][0] == '8':
            eight.append(digits[i][1])
        elif digits[i][0] == '9':
            nine.append(digits[i][1])

    sorted_digits.append(zero)
    sorted_digits.append(one)
    sorted_digits.append(two)
    sorted_digits.append(three)
    sorted_digits.append(four)
    sorted_digits.append(five)
    sorted_digits.append(six)
    sorted_digits.append(seven)
    sorted_digits.append(eight)
    sorted_digits.append(nine)

    return sorted_digits


def face_sort(faces, face_labels):

    zero = []
    one = []

    sorted_faces = []

    for i in range(451):

        if face_labels[i] == '0':
            zero.append(faces[i])
        elif face_labels[i] == '1':
            one.append(faces[i])

    sorted_faces.append(zero)
    sorted_faces.append(one)

    return sorted_faces


def print_digit_results(test_digits, sorted_digits, posteriors):

    right = [0 for x in range(10)]
    wrong = [0 for x in range(10)]

    confusion_matrix = [[0 for x in range(10)] for y in range(10)]

    for i in range(len(test_digits)):

        max_prob = -1000000000000
        max_num = 0

        for class_num in range(10):

            if posteriors[i][class_num] > max_prob:
                max_prob = posteriors[i][class_num]
                max_num = class_num

        if str(test_digits[i][0]) == str(max_num):
            right[int(test_digits[i][0])] = right[int(test_digits[i][0])] + 1
        else:
            wrong[int(test_digits[i][0])] = wrong[int(test_digits[i][0])] + 1

        confusion_matrix[int(test_digits[i][0])][max_num] = confusion_matrix[int(test_digits[i][0])][max_num] + 1

    print "\n","Confusion Matrix (1.1):"

    for i in range(10):
        for j in range(10):

            confusion_matrix[i][j] = confusion_matrix[i][j] / float(len(sorted_digits[i]))

        print confusion_matrix[i],"\n"

    print "zero accuracy (1.1):", right[0] / float(right[0] + wrong[0])
    print "one accuracy (1.1):", right[1] / float(right[1] + wrong[1])
    print "two accuracy (1.1):", right[2] / float(right[2] + wrong[2])
    print "three accuracy (1.1):", right[3] / float(right[3] + wrong[3])
    print "four accuracy (1.1):", right[4] / float(right[4] + wrong[4])
    print "five accuracy (1.1):", right[5] / float(right[5] + wrong[5])
    print "six accuracy (1.1):", right[6] / float(right[6] + wrong[6])
    print "seven accuracy (1.1):", right[7] / float(right[7] + wrong[7])
    print "eight accuracy (1.1):", right[8] / float(right[8] + wrong[8])
    print "nine accuracy (1.1):", right[9] / float(right[9] + wrong[9])

    total_right = 0
    total_wrong = 0

    for x in range(10):
        total_right = total_right + right[x]
        total_wrong = total_wrong + wrong[x]

    print "\n", "overall accuracy (1.1):", total_right / float(total_right + total_wrong)

    best_posteriors = [[-100000, []] for x in range(10)]
    worst_posteriors = [[100000, []] for x in range(10)]

    for i in range(len(test_digits)):

        for j in range(10):

            if posteriors[i][j] > best_posteriors[j][0]:
                best_posteriors[j][0] = posteriors[i][j]
                best_posteriors[j][1] = test_digits[i][1]

    for i in range(len(test_digits)):

        for j in range(10):

            if posteriors[i][j] < worst_posteriors[j][0] and str(test_digits[i][0]) == str(j):
                worst_posteriors[j][0] = posteriors[i][j]
                worst_posteriors[j][1] = test_digits[i][1]

    for i in range(10):
        print "\n"
        for x in range(32):
            print ""
            for y in range(32):
                digit_print.print_out(best_posteriors[i][1][x][y])


    for i in range(10):
        print "\n"
        for x in range(32):
            print ""
            for y in range(32):
                digit_print.print_out(worst_posteriors[i][1][x][y])


def print_face_results(test_face_labels, posteriors):

    right = [0 for x in range(2)]
    wrong = [0 for x in range(2)]

    confusion_matrix = [[0 for x in range(2)] for y in range(2)]

    label_count = [0 for x in range(2)]
    total_labels = len(test_face_labels)

    for i in range(len(test_face_labels)):

        max_prob = -1000000000000
        max_img = 0

        for class_img in range(2):

            if posteriors[i][class_img] > max_prob:
                max_prob = posteriors[i][class_img]
                max_img = class_img

        if str(test_face_labels[i]) == str(max_img):
            right[int(test_face_labels[i])] = right[int(test_face_labels[i])] + 1
        else:
            wrong[int(test_face_labels[i])] = wrong[int(test_face_labels[i])] + 1

        confusion_matrix[int(test_face_labels[i])][max_img] = confusion_matrix[int(test_face_labels[i])][max_img] + 1

    print "\nConfusion Matrix (1.3):"

    for x in range(total_labels):
        label_count[int(test_face_labels[x])] = label_count[int(test_face_labels[x])] + 1

    for i in range(2):
        for j in range(2):

            confusion_matrix[i][j] = confusion_matrix[i][j] / float(label_count[i])

        print confusion_matrix[i],"\n"

    print "face accuracy (1.3):", right[1] / float(right[1] + wrong[1])
    print "non face accuracy (1.3):", right[0] / float(right[0] + wrong[0])

    total_right = 0
    total_wrong = 0

    for x in range(2):
        total_right = total_right + right[x]
        total_wrong = total_wrong + wrong[x]

    print "\n","overall accuracy (1.3):", total_right / float(total_right + total_wrong)


def naiveBayes(train_digits, test_digits):

    sorted_digits = digit_sort(train_digits)

    k = 1	# constant for Laplace smoothing

    one_likelihoods = [[[0 for j in range(32)] for i in range(32)] for x in range(10)]
    zero_likelihoods = [[[0 for j in range(32)] for i in range(32)] for x in range(10)]

    priors = [0 for x in range(10)]

    for class_num in range(10):

        pixel = [[[] for j in range(32)] for i in range(32)]

        for exemplar in sorted_digits[class_num]:

            for i in range(32):
                for j in range(32):
                    pixel[i][j].append(exemplar[i][j])

        for i in range(32):
            for j in range(32):

                zero_count = 0
                one_count = 0

                for feature in pixel[i][j]:

                    if feature == '0':
                        zero_count = zero_count + 1
                    else:
                        one_count = one_count + 1

                zero_likelihoods[class_num][i][j] = (zero_count + k) / float((len(sorted_digits[class_num]) + (k * 2)))
                one_likelihoods[class_num][i][j] = (one_count + k) / float((len(sorted_digits[class_num]) + (k * 2)))

        priors[class_num] = len(sorted_digits[class_num]) / float(len(train_digits))

    posteriors = [[0 for x in range(10)] for y in range(len(test_digits))]
    count = -1

    for digit in test_digits:

        count = count + 1

        for class_num in range(10):

            posterior = math.log10(priors[class_num])

            for i in range(32):
                for j in range(32):

                    if (digit[1][i][j] == '0'):
                        posterior = posterior + math.log10(zero_likelihoods[class_num][i][j])
                    else:
                        posterior = posterior + math.log10(one_likelihoods[class_num][i][j])

            posteriors[count][class_num] = posterior

    #print_digit_results(test_digits, digit_sort(test_digits), posteriors)

    odds_ratio = [[0 for j in range(32)] for i in range(32)]

    # (2, 8), (5, 9), (4, 7) and (3, 9)

    for i in range(32):
        for j in range(32):
            odds_ratio[i][j] = math.log10(one_likelihoods[2][i][j] / float(one_likelihoods[8][i][j]))

    pixel_data = odds_ratio
    pixel_2D = np.reshape(pixel_data, (32, 32))
    plt.imshow(pixel_2D, cmap='jet', interpolation='nearest')
    name = 'OddsRatio_Class ' + str(0)
    plt.savefig(name)

    #### EXTRA CREDIT ####

    sorted_faces = face_sort(train_faces, train_face_labels)

    k = 1	# constant for Laplace smoothing

    one_likelihoods = [[[0 for j in range(60)] for i in range(70)] for x in range(2)]
    zero_likelihoods = [[[0 for j in range(60)] for i in range(70)] for x in range(2)]

    priors = [0 for x in range(2)]

    for class_img in range(2):

        pixel = [[[] for j in range(60)] for i in range(70)]

        for exemplar in sorted_faces[class_img]:

            for i in range(70):
                for j in range(60):
                    pixel[i][j].append(exemplar[i][j])

        for i in range(70):
            for j in range(60):

                zero_count = 0
                one_count = 0

                for feature in pixel[i][j]:

                    if feature == '#':
                        one_count = one_count + 1
                    else:
                        zero_count = zero_count + 1

                zero_likelihoods[class_img][i][j] = (zero_count + k) / float((len(sorted_faces[class_img]) + (k * 2)))
                one_likelihoods[class_img][i][j] = (one_count + k) / float((len(sorted_faces[class_img]) + (k * 2)))

        zeroes = 0
        ones = 0

        for label in train_face_labels:

            if label == '0':
                zeroes = zeroes + 1
            else:
                ones = ones + 1

        if class_img == 0:
            priors[0] = zeroes / float(len(train_face_labels))
        else:
            priors[1] = ones / float(len(train_face_labels))


    posteriors = [[0 for x in range(2)] for y in range(len(test_faces))]
    count = -1

    for face in test_faces:

        count = count + 1

        for class_img in range(2):

            posterior = math.log10(priors[class_img])

            for i in range(70):
                for j in range(60):

                    if (face[i][j] == '#'):
                        posterior = posterior + math.log10(one_likelihoods[class_img][i][j])
                    else:
                        posterior = posterior + math.log10(zero_likelihoods[class_img][i][j])

            posteriors[count][class_img] = posterior

    print_face_results(test_face_labels, posteriors)

def extractPixels():

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

    fd = open("facedatatrainlabels", "r")

    train_face_labels = []

    for i in range(451):

        face_data = []

        line = fd.readline()
        line = tuple(line)

        train_face_labels.append(line[0])


    fd = open("facedatatrain", "r")

    train_faces = []

    for i in range(451):

        face_data = []

        for j in range(70):

            line = fd.readline()
            line = tuple(line)
            face_data.append(line)

        train_faces.append(face_data)


    fd = open("facedatatestlabels", "r")

    test_face_labels = []

    for i in range(150):

        face_data = []

        line = fd.readline()
        line = tuple(line)

        test_face_labels.append(line[0])



    fd = open("facedatatest", "r")

    test_faces = []

    for i in range(150):

        face_data = []

        for j in range(70):

            line = fd.readline()
            line = tuple(line)
            face_data.append(line)

        test_faces.append(face_data)

    naiveBayes(train_digits, test_digits)

extractPixels()
