# 🎓 Convocation Face Recognition System

## 1. System Overview

This project implements a robust face recognition system primarily designed for **attendance and identity verification** during a formal event like a convocation ceremony. The core goal is to provide a fast and accurate mechanism to confirm the identity of attendees against a pre-registered database.

### Core Architecture

The system operates in two main phases:

1.  **Face Embeddings (Feature Extraction):** A pre-trained deep learning model (likely **FaceNet**) is used to process student photos and convert each face into a 128-dimensional vector, known as an **embedding**. These embeddings represent the unique features of each face.
2.  **Classification:** A separate, custom-trained Keras model (`facenet_embed_classifier.h5`) uses these unique embeddings to classify them and output the predicted identity of the student.

---

## 2. Project Components

| File/Folder Name | Description | Role in System |
| :--- | :--- | :--- |
| **`training_notebook.ipynb`** | Jupyter Notebook containing the full training workflow. | **Training & Data Prep:** Used to generate face embeddings and train the final classifier model. |
| **`facenet_embed_classifier.h5`** | The serialized Keras classification model. | **Prediction:** Contains the learned weights used by the system's interface to identify students from new face embeddings. |
| **`/interfaces`** | *(Placeholder for your interface folder)* | **User Interaction:** Contains all the scripts and files (e.g., Python code, UI elements) necessary for running the live camera or image verification system. |

---

## 3. Model Training Workflow

The model was prepared and trained using the steps documented in the **`training_notebook.ipynb`** notebook:

1.  **Data Loading:** Student images (the training database) are loaded.
2.  **Face Detection:** An algorithm (such as MTCNN) is used to locate and crop faces accurately from the loaded images.
3.  **Embedding Generation:** The cropped faces are passed through the FaceNet model to produce the 128-D embedding vectors.
    * The notebook code confirms that these embeddings are saved to a file (e.g., `flask_app_embeddings.pkl`).
4.  **Classifier Training:** The generated embeddings are used to train a simple, but effective, **Dense Neural Network** (the classifier) which learns to map the embedding vectors to the correct student ID.
5.  **Model Saving:** The trained classifier is saved as the **`facenet_embed_classifier.h5`** file for deployment in the main application.

---

## 4. Interface and Deployment

The live system is executed through the files within the interfaces folder. This typically involves:

* **Real-time Capture:** Using a webcam to capture the student's face.
* **Pipeline Execution:** Feeding the live face image through the same detection and embedding process.
* **Verification:** Loading the `facenet_embed_classifier.h5` model and feeding the live embedding to it for a final attendance or verification result.

---
