
**DISCLAIMER: This is a hobby project so do not expect everything to work :P**

### What is this?

Face recognition playard :)

### Dictionary

**embedding** - *just another representation of image (face). It is much easier to compare small vector than a huge multi-dimensional matrix (colored .jpg images)*

**class** - *many samples (embeddings) of each person (or any other ... object?)*

**classifier** - *helps You to find closest closest class for a samples (embedding) aka identifier*

**clustering** - *helps make sense of random embeddings. Input - collection of unlabeled embeddings. Output - labeled embeddings*

### How it works?

Collection of faces -> embed images -> compare random face embedding with stored embeddings


### Setup

- Install Python3 & requirements
- Download dlib models from https://github.com/davisking/dlib-models and store them under /models
    - dlib_face_recognition_resnet_model_v1.dat
    - mmod_human_face_detector.dat
    - shape_predictor_5_face_landmarks.dat
    - shape_predictor_68_face_landmarks.dat
- Create "database"
    - Make faces directory so that each person is stored /face_dir/person1/selfie1.jpg. NB! Add atleast 10 selfies to improve accuracy. For 100 persons, atleast 30 selfies are required.
    - Apply /utils/embed_pictures_dlib.py for the directory
    - Now You should have a face "database" /database/person1/selfie1.jpg and /database/person1/selfie1.emb
- Start /predictor/predict_from_video.py

### Utils

###### benchmark_classifiers.py

For selecting best classifier and their configuration for Your embeddings. If best combination is found, feel free to apply them for Your indentifier.

**NB!** I will try to add embedding generators for different models later on

###### benchmark_clusertings.py

Clustering can help You to create datasets automatically. As the results are not 100% valid, one needs to find perfect parameters and clustering methods for that.

###### combine_embeddings.py

Make classes of unidentified samples. **NB! Also keep .jpg images in order to validate the results afterwards.**

###### embed_pictures_dlib.py

Create comparable data from dataset
