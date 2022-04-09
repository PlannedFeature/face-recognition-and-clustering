from sklearn import metrics
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
import os

''' Private variables
------------------------------------------------------------------
'''
FACE_DATABASE_DIR = ""
SAMPLES_PER_PERSON = 30
AMOUNT_OF_VALIDATION_SAMPLES = 6  # 20% for validition samples for known people
AMOUNT_OF_KNOWN_PERSONS = 131    # 80% of dataset is known, rest will be unknown
AMOUNT_OF_PROCESSING_SAMPLES = SAMPLES_PER_PERSON - AMOUNT_OF_VALIDATION_SAMPLES
THRESHOLD = 0.475
MAX_PEOPLE_TO_INCLUDE = 100
LIMIT_PEOPLE = False

# training inputs will be only known people
training_inputs = []
training_outputs = []
# validation inputs contaion known and unknown people
validation_inputs = []
validation_outputs = []

print("Reading data to memory")
# Load faces to memory
for root, dirs, files in os.walk(FACE_DATABASE_DIR):
    number_of_folders_read = 0
    for folder in dirs:
        folder_path = os.path.join(root, folder)
        number_of_samples_per_person = 0
        for emb in os.listdir(folder_path):
            if emb.endswith('.emb'):
                emb_file_name = (os.path.join(folder_path, emb))
                img_embeddings = []
                with open(emb_file_name, 'r') as f:
                    img_embeddings = [float(line.rstrip('\n')) for line in f]
                    if (number_of_folders_read < AMOUNT_OF_KNOWN_PERSONS):
                        if number_of_samples_per_person < AMOUNT_OF_PROCESSING_SAMPLES:
                            training_inputs.append(img_embeddings)
                            training_outputs.append(int(folder))
                        else:
                            validation_inputs.append(img_embeddings)
                            validation_outputs.append(int(folder))
                    else:
                        validation_inputs.append(img_embeddings)
                        # create unknown people for dataset
                        validation_outputs.append(-1)
                    number_of_samples_per_person += 1
        number_of_folders_read += 1
        if LIMIT_PEOPLE and number_of_folders_read > LIMIT_PEOPLE:
            print("Not loading whole database, because of limitations")
            break

total_samples = len(training_inputs) + len(validation_inputs)

print("Train test ration: " + str(float(len(validation_inputs) / total_samples)))
print("Data loaded, starting to process")

print("KNR classifier results")
images_for_training = 0.0  # percent
while images_for_training < 0.9:
    print("Images for training: " + str(AMOUNT_OF_PROCESSING_SAMPLES -
          float(AMOUNT_OF_PROCESSING_SAMPLES * images_for_training)))
    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(
        training_inputs, training_outputs, test_size=images_for_training)
    best_accuracy = {
        'radius': None,
        'acc': 0.0,
        'neighbours': 0
    }
    radius = 0.1
    while radius < 1.0:
        knn_clf = RadiusNeighborsClassifier(
            radius=radius, outlier_label=-1, algorithm='ball_tree', n_jobs=-1)
        knn_clf.fit(train_inputs, train_outputs)
        predictions = knn_clf.predict(validation_inputs)
        neighbors = knn_clf.radius_neighbors(validation_inputs)
        neighbourhood_sizes = [len(i) for i in neighbors[0]]
        accuracy = metrics.accuracy_score(validation_outputs, predictions)
        #print("Radius: " + str(radius) + ", accuracy: " + str(accuracy))
        if (accuracy > best_accuracy['acc']):
            best_accuracy['acc'] = accuracy
            best_accuracy['radius'] = radius
            # we only want to know neigbours of the known people
            best_accuracy['neighbours'] = np.average(
                neighbourhood_sizes[:AMOUNT_OF_KNOWN_PERSONS])
        print(str(accuracy) + ":" + str(radius) + ":" +
              str(np.average(neighbourhood_sizes[:AMOUNT_OF_KNOWN_PERSONS])))
        #print("Number of neighbours: " + str(amount_of_neighbours) + ", accuracy: " + str(accuracy))
        radius += 0.05
    #print(str(best_accuracy['acc']) + ":" + str(best_accuracy['radius']) + ":" + str(best_accuracy['neighbours']))
    images_for_training += 0.1

print("KNN classifier results")
images_for_training = 0.0  # percent
while images_for_training < 0.9:
    print("Images for training: " + str(AMOUNT_OF_PROCESSING_SAMPLES -
          float(AMOUNT_OF_PROCESSING_SAMPLES * images_for_training)))
    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(
        training_inputs, training_outputs, test_size=images_for_training)
    best_accuracy = {
        'n': None,
        'acc': 0.0
    }
    for amount_of_neighbours in range(1, 30):
        knn_clf = neighbors.KNeighborsClassifier(
            n_neighbors=4, algorithm='ball_tree', weights='distance', n_jobs=-1)
        knn_clf.fit(train_inputs, train_outputs)
        distances = knn_clf.kneighbors(
            validation_inputs, n_neighbors=4, return_distance=True)
        is_close_enough = [distances[0][i][0] <=
                           THRESHOLD for i in range(len(validation_inputs))]
        predictions = knn_clf.predict(validation_inputs)
        probabilities = knn_clf.predict_proba(validation_inputs)
        dominating_class_indexes = np.argmax(probabilities, axis=1)
        dominating_class_probabilities = probabilities[np.arange(
            len(dominating_class_indexes)), dominating_class_indexes]
        # filter out unknown people
        outputs_after_filtering = [int(pred) if is_close_enough[i] and dominating_class_probabilities[i]
                                   == 1. else -1 for i, pred in enumerate(predictions)]
        # print(outputs_after_filtering)
        accuracy = metrics.accuracy_score(
            validation_outputs, outputs_after_filtering)
        if (accuracy > best_accuracy['acc']):
            best_accuracy['acc'] = accuracy
            best_accuracy['n'] = amount_of_neighbours
        #print("Number of neighbours: " + str(amount_of_neighbours) + ", accuracy: " + str(accuracy))

    print(best_accuracy)
    images_for_training += 0.1


# Test different variation of SVM
print("SVM classifier results")
images_for_training = 0.0  # percent
while images_for_training < 0.9:
    # Select how many if training images are we giving for classifier
    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(
        training_inputs, training_outputs, test_size=images_for_training)
    print("Images for training: " + str(AMOUNT_OF_PROCESSING_SAMPLES -
          float(AMOUNT_OF_PROCESSING_SAMPLES * images_for_training)))
    svc_classifier = SVC(C=3, kernel='linear',
                         probability=True, decision_function_shape='ovr')
    svc_classifier.fit(train_inputs, train_outputs)
    probabilities = svc_classifier.predict_proba(validation_inputs)
    dominating_class_indexes = np.argmax(probabilities, axis=1)
    dominating_class_probabilities = probabilities[np.arange(
        len(dominating_class_indexes)), best_cdominating_class_indexeslass_indices]
    # print(dominating_class_probabilities)
    predictions = svc_classifier.predict(validation_inputs)
    outputs_after_filtering = [-1 if proba < 0.1 else predictions[i] for i, proba in enumerate(
        dominating_class_probabilities)]  # if confidence is low, then mark is as unknown
    print("Accuracy:", metrics.accuracy_score(
        validation_outputs, outputs_after_filtering))
    images_for_training += 0.1
