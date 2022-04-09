from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import dlib
import numpy as np
import os
import shutil
import time
import uuid

CLUSTERED_VIDEOS = ""
RESULT_DIR_CLUSTERING_ALG = ""
THRESHOLD = 0.475
CLASS_MEMBERS = 50
REQUIRED_SAMPLES_PER_ELEVATOR_CYCLE = 5

all_classes = dict()
for video in os.listdir(CLUSTERED_VIDEOS):
    sub_dir = os.path.join(CLUSTERED_VIDEOS, video)
    for face_dir in os.listdir(sub_dir):
        face_sub_dir = os.path.join(sub_dir, face_dir)
        number_of_files = len(os.listdir(face_sub_dir))
        # too little samples per video, probably mismatch
        if (number_of_files / 2) < REQUIRED_SAMPLES_PER_ELEVATOR_CYCLE:
            continue
        for face_file in os.listdir(face_sub_dir):
            file_type = face_file[-3:]
            if file_type == "jpg":
                # get picture embedding
                embedding_file_path = os.path.join(
                    face_sub_dir, face_file[:-3] + "emb")
                embedding = []
                with open(embedding_file_path, 'r') as f:
                    embedding = [float(line.rstrip('\n')) for line in f]
                # create processable node
                tmp_class_id = uuid.uuid1()
                node = {
                    "initialized_class": tmp_class_id,
                    "label": face_file,
                    "embedding": embedding,
                    "embedding_path": embedding_file_path,
                    "picture_path": os.path.join(face_sub_dir, face_file)
                }

                if tmp_class_id not in all_classes:
                    all_classes[tmp_class_id] = dict()
                    all_classes[tmp_class_id]["nodes"] = []
                    all_classes[tmp_class_id]["origin"] = face_sub_dir

                all_classes[tmp_class_id]["nodes"].append(node)

print("Embeddings imported")
if os.path.exists(RESULT_DIR_CLUSTERING_ALG):  # result dir cleaning
    shutil.rmtree(RESULT_DIR_CLUSTERING_ALG)

os.mkdir(RESULT_DIR_CLUSTERING_ALG)  # directory for embeddings

all_embeddings = []
all_pictures_paths = []
all_embedding_paths = []

for face_class in all_classes:
    for face in all_classes[face_class]["nodes"]:
        all_embeddings.append(dlib.vector(face["embedding"]))
        all_pictures_paths.append(face["picture_path"])
        all_embedding_paths.append(face["embedding_path"])

numpy_all_embeddings = np.array(all_embeddings)
chinese_whipers_started = time.time()
chinese_whisper_classes = dlib.chinese_whispers_clustering(
    all_embeddings, THRESHOLD)
chinese_whipers_ended = time.time()
dlib_chinese_whispers_number_of_unique_classes = len(
    set(chinese_whisper_classes))
print("Finished clustering")

processed_labels = []
for i, label in enumerate(chinese_whisper_classes):
    if label in processed_labels:  # we have already processed that labels
        continue
    processed_labels.append(label)
    face_indexes = []
    j = i
    # find all indexes, which have same label
    while j < len(chinese_whisper_classes):
        if chinese_whisper_classes[j] == label:
            face_indexes.append(j)
        j += 1
    if len(face_indexes) < CLASS_MEMBERS:  # too little examples anyways
        continue
    embeddings_with_same_label = numpy_all_embeddings[face_indexes]
    # calculate the center node
    neighbourhood_police = KMeans(n_clusters=1, n_jobs=-1)
    neighbourhood_of_the_class = neighbourhood_police.fit(
        embeddings_with_same_label)
    center_node = (neighbourhood_police.cluster_centers_)[0]
    # get the closest matches to center node
    neighbourhood = NearestNeighbors(metric="euclidean", n_jobs=-1)
    neighbourhood.fit(embeddings_with_same_label)
    closest_matches_from_center = neighbourhood.kneighbors(
        [center_node], CLASS_MEMBERS, return_distance=False)[0]
    closest_matches_from_center.sort()
    k = 0
    while k < len(closest_matches_from_center):
        idx = face_indexes[k]
        class_destination_dir = os.path.join(
            RESULT_DIR_CLUSTERING_ALG, str(label))
        if not os.path.exists(class_destination_dir):
            os.mkdir(class_destination_dir)

        shutil.copy(all_pictures_paths[idx],
                    os.path.join(class_destination_dir))
        shutil.copy(all_embedding_paths[idx],
                    os.path.join(class_destination_dir))
        k += 1


print("Chinese Whispers took time " +
      str(chinese_whipers_ended - chinese_whipers_started) + "s")
print("Class reduced to " + str(dlib_chinese_whispers_number_of_unique_classes) +
      " from: " + str(len(all_classes)))
print("Total embeddings: " + str(len(all_embeddings)))
