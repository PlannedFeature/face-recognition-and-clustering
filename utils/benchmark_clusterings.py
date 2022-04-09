from sklearn.cluster import DBSCAN, MeanShift
import dlib
import os
import random
import time
import numpy as np

''' Private variables
------------------------------------------------------------------
'''
OPENFACE_EMB_DIR = ""
DLIB_EMB_DIR = ""
FACENET_EMB_DIR = ""
BENCMARKS_FOR_X_PEOPLE = [10, 100, 2000, 1002000, 5000]
IRERATIONS_PER_PEOPEL_NR = 20
MAX_THRESHOLD = 1.5


def class_diff(correct_classes, predicted_classes):
    total_elements = len(correct_classes)
    seen_class_ids = []
    mismatches = 0
    accumulator = 0
    current_class = -2
    i = 0  # index of correct classes
    while i < len(correct_classes):
        beginning_of_class = i
        if current_class != correct_classes[i]:  # class changed
            current_class = correct_classes[i]
            accumulator = 0
            while i < len(correct_classes):  # count class instances
                if correct_classes[i] != current_class:
                    break
                else:
                    accumulator += 1
                    i += 1
            # one class has been ended, check if this predicted had same object pattern
            j = beginning_of_class
            current_label_in_predicted = predicted_classes[beginning_of_class]

            while j < (beginning_of_class + accumulator):
                # faces of this classes should have been already ended
                if predicted_classes[j] in seen_class_ids:
                    mismatches += 1
                elif current_label_in_predicted != predicted_classes[j]:
                    mismatches += 1
                j += 1
            seen_class_ids.append(current_label_in_predicted)

    return float("{0:.2f}".format(100 - 100*(float(mismatches) / float(total_elements))))


def find_averages(array_of_accuracy_obejcts):
    average_treshold = np.mean([x["threshold"]
                               for x in array_of_accuracy_obejcts])
    average_accuracy = np.mean([x["accuracy"]
                               for x in array_of_accuracy_obejcts])

    return average_accuracy, average_treshold


for current_batch_count in BENCMARKS_FOR_X_PEOPLE:  # benchmark 10,100,100
    # Average results for batch
    facenet_db_scan_avarage_results = []
    facenet_mean_shift_avarage_results = []
    facenet_chinese_whispers_avarage_results = []
    dlib_db_scan_avarage_results = []
    dlib_mean_shift_avarage_results = []
    dlib_chinese_whispers_avarage_results = []
    openface_db_scan_avarage_results = []
    openface_mean_shift_avarage_results = []
    openface_chinese_whispers_avarage_results = []

    # Timings are part of batch
    dbscan_timings = []
    mean_shift_timings = []
    chinese_whispers_timings = []

    iterators_finished = 0
    while iterators_finished < IRERATIONS_PER_PEOPEL_NR:
        # Results for inidiviual iteration
        facenet_db_scan_results = {}
        facenet_mean_shift_results = {}
        facenet_chinese_whispers_results = {}
        dlib_db_scan_results = {}
        dlib_mean_shift_results = {}
        dlib_chinese_whispers_results = {}
        openface_db_scan_results = {}
        openface_mean_shift_results = {}
        openface_chinese_whispers_results = {}

        print("Starting iteration " + str(iterators_finished + 1) +
              " of " + str(IRERATIONS_PER_PEOPEL_NR))

        # Embeddings of random people
        facenet_embeddings_random = []
        facenet_class_ids = []
        dlib_embeddings_random = []
        dlib_class_ids = []
        openface_embeddings_random = []
        openface_class_ids = []

        random_people_indexes = [random.randint(1, 5000) for _ in range(
            current_batch_count)]  # random people to take

        # Openface embeddings
        for root, dirs, files in os.walk(OPENFACE_EMB_DIR):
            class_id = 0
            for i in random_people_indexes:
                rand_person_folder = os.path.join(root, dirs[i])
                for emb in os.listdir(rand_person_folder):
                    emb_file_name = os.path.join(rand_person_folder, emb)
                    with open(emb_file_name, 'r') as f:
                        img_embeddings = [float(line.rstrip('\n'))
                                          for line in f]
                        openface_embeddings_random.append(img_embeddings)
                        openface_class_ids.append(class_id)

                class_id += 1
            break

        # Facenet embeddings
        for root, dirs, files in os.walk(FACENET_EMB_DIR):
            class_id = 0
            for i in random_people_indexes:
                rand_person_folder = os.path.join(root, dirs[i])
                for emb in os.listdir(rand_person_folder):
                    emb_file_name = os.path.join(rand_person_folder, emb)
                    with open(emb_file_name, 'r') as f:
                        img_embeddings = [float(line.rstrip('\n'))
                                          for line in f]
                        facenet_embeddings_random.append(img_embeddings)
                        facenet_class_ids.append(class_id)
                class_id += 1
            break

        # Dlib embeddings
        for root, dirs, files in os.walk(DLIB_EMB_DIR):
            class_id = 0
            for i in random_people_indexes:
                rand_person_folder = os.path.join(root, dirs[i])
                for emb in os.listdir(rand_person_folder):
                    emb_file_name = os.path.join(rand_person_folder, emb)
                    with open(emb_file_name, 'r') as f:
                        img_embeddings = [float(line.rstrip('\n'))
                                          for line in f]
                        dlib_embeddings_random.append(img_embeddings)
                        dlib_class_ids.append(class_id)
                class_id += 1
            break

        threshold = 0.1

        while threshold < MAX_THRESHOLD:  # change thresholds
            percentage = (threshold / MAX_THRESHOLD) * 100
            print(str(percentage) + "% done..")
            threshold = round(threshold, 2)
            for i in range(3):  # process each algorithm individually for timings
                # Facenet
                if i == 0:
                    for j in range(3):
                        if j == 0:
                            facenet_dbscan_cluster = DBSCAN(
                                eps=threshold, min_samples=1, metric="euclidean", n_jobs=-1)
                            start = time.time()
                            facenet_dbscan_cluster.fit(
                                facenet_embeddings_random)
                            stop = time.time()
                            dbscan_timings.append(stop - start)
                            facenet_dbscan_cluster_cluster_labels_with_random = facenet_dbscan_cluster.labels_
                            facenet_db_scan_results[threshold] = class_diff(
                                facenet_class_ids, facenet_dbscan_cluster_cluster_labels_with_random)
                        if j == 1:
                            facenet_mean_shift_cluster = MeanShift(
                                bandwidth=threshold)
                            start = time.time()
                            facenet_mean_shift_cluster.fit(
                                facenet_embeddings_random)
                            stop = time.time()
                            mean_shift_timings.append(stop - start)
                            facenet_mean_shift_cluster_labels_with_random = facenet_mean_shift_cluster.labels_
                            facenet_mean_shift_results[threshold] = class_diff(
                                facenet_class_ids, facenet_mean_shift_cluster_labels_with_random)
                        if j == 2:
                            facenet_chinese_whispers_cluster_labels_with_random = dlib.chinese_whispers_clustering(
                                [dlib.vector(l) for l in facenet_embeddings_random], threshold)
                            facenet_chinese_whispers_results[threshold] = class_diff(
                                facenet_class_ids, facenet_chinese_whispers_cluster_labels_with_random)

                if i == 1:
                    # Dlib
                    for j in range(3):
                        if j == 0:
                            dlib_dbscan_cluster = DBSCAN(
                                eps=threshold, min_samples=1, metric="euclidean", n_jobs=-1)
                            start = time.time()
                            dlib_dbscan_cluster.fit(dlib_embeddings_random)
                            stop = time.time()
                            dbscan_timings.append(stop - start)
                            dlib_dbscan_cluster_cluster_labels_with_random = dlib_dbscan_cluster.labels_
                            dlib_db_scan_results[threshold] = class_diff(
                                dlib_class_ids, dlib_dbscan_cluster_cluster_labels_with_random)
                        if j == 1:
                            dlib_mean_shift_cluster = MeanShift(
                                bandwidth=threshold)
                            dlib_mean_shift_cluster.fit(dlib_embeddings_random)
                            dlib_mean_shift_cluster_labels_with_random = dlib_mean_shift_cluster.labels_
                            dlib_mean_shift_results[threshold] = class_diff(
                                dlib_class_ids, dlib_mean_shift_cluster_labels_with_random)
                        if j == 2:
                            start = time.time()
                            dlib_chinese_whispers_cluster_labels_with_random = dlib.chinese_whispers_clustering(
                                [dlib.vector(l) for l in dlib_embeddings_random], threshold)
                            stop = time.time()
                            chinese_whispers_timings.append(stop - start)
                            dlib_chinese_whispers_results[threshold] = class_diff(
                                dlib_class_ids, dlib_chinese_whispers_cluster_labels_with_random)

                if i == 2:
                    for j in range(3):
                        if j == 0:
                            # Openface
                            openface_dbscan_cluster = DBSCAN(
                                eps=threshold, min_samples=1, metric="euclidean", n_jobs=-1)
                            openface_dbscan_cluster.fit(
                                openface_embeddings_random)
                            openface_dbscan_cluster_cluster_labels_with_random = openface_dbscan_cluster.labels_
                            openface_db_scan_results[threshold] = class_diff(
                                openface_class_ids, openface_dbscan_cluster_cluster_labels_with_random)
                        if j == 1:
                            openface_chinese_whispers_cluster_labels_with_random = dlib.chinese_whispers_clustering(
                                [dlib.vector(l) for l in openface_embeddings_random], threshold)
                            openface_chinese_whispers_results[threshold] = class_diff(
                                openface_class_ids, openface_chinese_whispers_cluster_labels_with_random)
                        if j == 2:
                            openface_mean_shift_cluster = MeanShift(
                                bandwidth=threshold)
                            openface_mean_shift_cluster.fit(
                                openface_embeddings_random)
                            openface_mean_shift_cluster_labels_with_random = openface_mean_shift_cluster.labels_
                            openface_mean_shift_results[threshold] = class_diff(
                                openface_class_ids, openface_mean_shift_cluster_labels_with_random)

            threshold += 0.05

        threshold, accuracy = sorted(
            facenet_db_scan_results.items(), key=lambda v: v[1], reverse=True)[0]
        accuracy_obj = {
            'threshold': threshold,
            'accuracy': accuracy
        }
        facenet_db_scan_avarage_results.append(accuracy_obj)

        threshold, accuracy = sorted(
            facenet_mean_shift_results.items(), key=lambda v: v[1], reverse=True)[0]
        accuracy_obj = {
            'threshold': threshold,
            'accuracy': accuracy
        }
        facenet_mean_shift_avarage_results.append(accuracy_obj)

        threshold, accuracy = sorted(
            facenet_chinese_whispers_results.items(), key=lambda v: v[1], reverse=True)[0]
        accuracy_obj = {
            'threshold': threshold,
            'accuracy': accuracy
        }
        facenet_chinese_whispers_avarage_results.append(accuracy_obj)

        threshold, accuracy = sorted(
            dlib_db_scan_results.items(), key=lambda v: v[1], reverse=True)[0]
        accuracy_obj = {
            'threshold': threshold,
            'accuracy': accuracy
        }
        dlib_db_scan_avarage_results.append(accuracy_obj)

        threshold, accuracy = sorted(
            dlib_mean_shift_results.items(), key=lambda v: v[1], reverse=True)[0]
        accuracy_obj = {
            'threshold': threshold,
            'accuracy': accuracy
        }
        dlib_mean_shift_avarage_results.append(accuracy_obj)

        threshold, accuracy = sorted(
            dlib_chinese_whispers_results.items(), key=lambda v: v[1], reverse=True)[0]
        accuracy_obj = {
            'threshold': threshold,
            'accuracy': accuracy
        }
        dlib_chinese_whispers_avarage_results.append(accuracy_obj)

        threshold, accuracy = sorted(
            openface_db_scan_results.items(), key=lambda v: v[1], reverse=True)[0]
        accuracy_obj = {
            'threshold': threshold,
            'accuracy': accuracy
        }
        openface_db_scan_avarage_results.append(accuracy_obj)

        threshold, accuracy = sorted(
            openface_mean_shift_results.items(), key=lambda v: v[1], reverse=True)[0]
        accuracy_obj = {
            'threshold': threshold,
            'accuracy': accuracy
        }
        openface_mean_shift_avarage_results.append(accuracy_obj)

        threshold, accuracy = sorted(
            openface_chinese_whispers_results.items(), key=lambda v: v[1], reverse=True)[0]
        accuracy_obj = {
            'threshold': threshold,
            'accuracy': accuracy
        }
        openface_chinese_whispers_avarage_results.append(accuracy_obj)

        iterators_finished += 1

    # iterations have been finished
    print("--- RESULTS FOR " + str(current_batch_count) + " PEOPLE ---")
    threshold, accuracy = find_averages(facenet_db_scan_avarage_results)
    print("Facenet->DBSCAN: " + str(accuracy) + ":" + str(threshold))
    threshold, accuracy = find_averages(
        facenet_chinese_whispers_avarage_results)
    print("Facenet->CW: " + str(accuracy) + ":" + str(threshold))
    threshold, accuracy = find_averages(facenet_mean_shift_avarage_results)
    print("Facenet->Mean-Shift: " + str(accuracy) + ":" + str(threshold))
    threshold, accuracy = find_averages(dlib_chinese_whispers_avarage_results)
    print("Dlib->CW: " + str(accuracy) + ":" + str(threshold))
    threshold, accuracy = find_averages(dlib_mean_shift_avarage_results)
    print("Dlib->Mean-Shift: " + str(accuracy) + ":" + str(threshold))
    threshold, accuracy = find_averages(dlib_db_scan_avarage_results)
    print("Dlib->DBSCAN: " + str(accuracy) + ":" + str(threshold))
    threshold, accuracy = find_averages(openface_db_scan_avarage_results)
    print("OpenFace->DBSCAN: " + str(accuracy) + ":" + str(threshold))
    threshold, accuracy = find_averages(openface_mean_shift_avarage_results)
    print("OpenFace->Mean-Shift: " + str(accuracy) + ":" + str(threshold))
    threshold, accuracy = find_averages(
        openface_chinese_whispers_avarage_results)
    print("OpenFace->CW: " + str(accuracy) + ":" + str(threshold))

    print("Timings with " + str(len(dlib_embeddings_random)) + " embeddings:")
    print("DBSCAN:" + str(np.mean(dbscan_timings)))
    print("Mean-Shift:" + str(np.mean(mean_shift_timings)))
    print("Chinese whispers:" + str(np.mean(chinese_whispers_timings)))
