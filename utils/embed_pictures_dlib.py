import cv2
import dlib
import os
import shutil
''' Private variables
------------------------------------------------------------------
'''
# dir, where the imput images are like:
# /faces
#   /person1
#       /selfie1.jpg
#       /selfie2.jpg
FACES_DIR = ""
# Output dir of the cropped faces and embeddings
EMDEDDED_FACES_DIR = ""
# Location of shape_predictor_68_face_landmarks.dat
SHAPE_PREDICTOR = ""
# Location of dlib_face_recognition_resnet_model_v1.dat
FEATURE_EXTRACTOR = ""

# Load dlib models
print("Loading NN")
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
feature_extractor = dlib.face_recognition_model_v1(FEATURE_EXTRACTOR)

print("Processing images")
embedded_images = {}

for root, dirs, files in os.walk(FACES_DIR):
    for folder in dirs:
        embedded_images[folder] = []
        folder_path = os.path.join(root, folder)
        for picture in os.listdir(folder_path):
            print("Processing " + picture)
            picture_path = os.path.join(folder_path, picture)
            image = dlib.load_rgb_image(picture_path)
            dets = face_detector(image, 2)
            for k, rect in enumerate(dets):
                shape = shape_predictor(image, rect)
                cropped_face = image[rect.top():rect.bottom(),
                                     rect.left():rect.right(), :]
                feature_vector = feature_extractor.compute_face_descriptor(
                    image, shape, 100)
                embedded_images[folder].append({
                    "vector": feature_vector,
                    "image": cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB),
                })

print("Writing embeddings to files")
new_root_path = os.path.join(os.getcwd(), EMDEDDED_FACES_DIR)
if not os.path.exists(new_root_path):
    os.mkdir(new_root_path)
for name, metadata in embedded_images.items():
    new_face_path = os.path.join(new_root_path, name)
    if not os.path.exists(new_face_path):
        os.mkdir(new_face_path)
    for i, node in enumerate(metadata):
        emb_file_name = os.path.join(new_face_path, str(i) + ".emb")
        pic_file_name = os.path.join(new_face_path, str(i) + ".jpg")
        cv2.imwrite(pic_file_name, node["image"])
        with open(emb_file_name, 'w') as f:
            for param in node["vector"]:
                f.write(str(param) + '\n')
