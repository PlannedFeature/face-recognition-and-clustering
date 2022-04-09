from sklearn.neighbors import RadiusNeighborsClassifier
import cv2
import dlib
import numpy as np
import os
import copy


class Identifier:
    '''
    * @brief  Constructor of Identifier object
    * @param  Float threshold - similarity threshold between faces
    * @param  Int min_matches - number of matches required inside radious
    * @param  String known_people_dir - path of the face database used for identification
    * @param  String shape_predictor_model_path - path of shape_predictor_68_face_landmarks.dat
    * @param  String resnet_model_path - path of dlib_face_recognition_resnet_model_v1.dat
    '''

    def __init__(self, threshold, known_people_dir, shape_predictor_model_path, resnet_model_path, min_matches=1):
        self.__min_matches = min_matches
        print("Ininitialising neural networks")
        self.__face_detector = dlib.get_frontal_face_detector()
        self.__shape_predictor = dlib.shape_predictor(
            shape_predictor_model_path)
        self.__feature_extractor = dlib.face_recognition_model_v1(
            resnet_model_path)
        print("Loading known people to memory")
        x = []
        y = []
        # can also read from database
        for root, dirs in os.walk(os.path.join(os.getcwd(), known_people_dir)):
            for folder in dirs:
                print(folder)
                folder_path = os.path.join(root, folder)
                for emb in os.listdir(folder_path):
                    emb_file_name = (os.path.join(folder_path, emb))
                    if emb_file_name.endswith('.emb'):
                        img_embeddings = []
                        with open(emb_file_name, 'r') as f:
                            img_embeddings = [
                                float(line.rstrip('\n')) for line in f]
                            x.append(img_embeddings)
                            y.append(folder)
        self.__classifier = RadiusNeighborsClassifier(
            radius=threshold, outlier_label='Unknown', algorithm='ball_tree', n_jobs=-1)
        self.__classifier.fit(x, y)
        print("Using CUDA: " + str(dlib.DLIB_USE_CUDA))

    '''
    * @brief  Idenfifies person in the frame
    * @param  Frame frame - processable frame
    * @return - Dict of processed results:
        {
            "processed_frame"   - frame with predictions
            "face_boxes"        - Array of boundary_boxes of detected faces
            "feature_vectors"   - Features extracted from faces
            "predictions"       - Array identified people
            "faces"             - Array cropped faces of people
        }
    '''

    def identify(self, frame):
        processed_frame = copy.deepcopy(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_arr = np.asarray(frame.shape)[0:2]
        boundary_boxes = self.__face_detector(rgb_frame, 1)
        faces = []
        predictions = []
        feature_vectors = []
        for k, rect in enumerate(boundary_boxes):
            # partial face
            if rect.top() < 0 or rect.bottom() > img_arr[1] or rect.left() < 0 or rect.right() > img_arr[0]:
                continue
            cropped_face = frame[max(0, rect.top()):min(rect.bottom(), img_arr[1]), max(
                0, rect.left()):min(rect.right(), img_arr[0]), :]
            shape = self.__shape_predictor(rgb_frame, rect)
            feature_vector = self.__feature_extractor.compute_face_descriptor(
                rgb_frame, shape, 1)
            matches_count = len(
                self.__classifier.radius_neighbors([feature_vector])[0])
            prediction = self.__classifier.predict([feature_vector])[
                0] if matches_count >= self.__min_matches else 'Unknown'
            #print("prediction ", prediction)
            predictions.append(prediction)
            #print("predictions ", predictions)
            faces.append(cropped_face)
            feature_vectors.append(feature_vector)
            cv2.rectangle(processed_frame, (rect.left(), rect.top()),
                          (rect.right(), rect.bottom()), (0, 255, 0), 2)
            cv2.putText(processed_frame, prediction, (rect.left(
            ) + 6, rect.bottom() - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 1)

        return {
            "processed_frame": processed_frame,
            "face_boxes": boundary_boxes,
            "feature_vectors": feature_vectors,
            "predictions": predictions,
            "faces": faces
        }
