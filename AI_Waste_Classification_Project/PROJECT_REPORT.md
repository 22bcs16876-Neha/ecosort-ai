# Project Report: AI Waste Classification & Smart Recycling System

## 1. Abstract
This project develops an image-based waste classification system using machine learning and computer vision. A user provides a photograph of a waste item, and a trained deep-learning model predicts whether it belongs to cardboard, glass, metal, paper, plastic or trash. The application also provides general recycling guidance. The objective is to demonstrate how AI can support waste segregation and environmental awareness.

## 2. Problem statement
Incorrect waste segregation makes recycling difficult. Manual sorting can be slow and inconsistent. This project explores whether an image-classification model can assist people in identifying common waste categories.

## 3. Objectives
- Understand image classification and convolutional neural networks.
- Prepare and explore a labelled image dataset.
- Train and evaluate a transfer-learning model.
- Build an interactive application for image upload and camera input.
- Understand model limitations and responsible use.

## 4. Dataset
The project uses TrashNet, a public educational dataset with six categories: cardboard, glass, metal, paper, plastic and trash. The resized collection is described as 2,527 images in the original project README. The dataset is not perfectly balanced; the trash class has fewer images. A stratified split is used to preserve class proportions.

Dataset source: https://github.com/garythung/trashnet
Dataset mirror: https://huggingface.co/datasets/garythung/trashnet

## 5. Methodology
1. Download and inspect the labelled images.
2. Split images into 70% training, 15% validation and 15% testing sets.
3. Resize images to 224 × 224 pixels.
4. Apply augmentation such as horizontal flipping, small rotations and zoom.
5. Use MobileNetV2 pretrained on ImageNet as a feature extractor.
6. Add global average pooling, dropout and a six-class softmax output layer.
7. Train the classification head, then fine-tune the final backbone layers with a small learning rate.
8. Evaluate on the held-out test set.

## 6. How the model works
A CNN learns visual patterns from images. Early layers detect simple features such as edges and textures; deeper layers combine these into more meaningful patterns. Transfer learning starts with features learned from a large image collection and adapts them to the waste categories. Softmax converts the output scores into class probabilities.

## 7. Evaluation
After training, record the actual values from `outputs/test_metrics.json`.
- Test accuracy: __________________
- Test loss: __________________
- Best-performing category: __________________
- Category needing improvement: __________________

Use the confusion matrix and classification report to discuss precision, recall and F1-score. Do not insert made-up accuracy values.

## 8. Application
The Streamlit interface accepts a JPG/PNG upload or a camera capture. It displays the predicted class, confidence, top-three predictions and general disposal guidance.

## 9. Limitations
- The dataset is relatively small and has an uneven class distribution.
- Many images have simple backgrounds unlike real bins or streets.
- A single image may contain multiple waste objects.
- Material type can be difficult to identify from appearance alone.
- Recycling rules differ by city and waste facility.
- The model is not a certified waste or safety decision system.

## 10. Future scope
- Add organic/biodegradable, e-waste, textile and battery categories using a suitable labelled dataset.
- Collect local images from the school campus with consent and correct labels.
- Add object detection to identify multiple items in one frame.
- Connect the prediction to a smart bin prototype using a microcontroller and servo motor (with teacher supervision).
- Create a dashboard to count categories and estimate recycling trends.

## 11. Conclusion
The project demonstrates an end-to-end machine-learning workflow: dataset preparation, transfer learning, model evaluation and a user-facing application. It provides a practical introduction to AI for environmental sustainability while showing why testing, data quality and responsible interpretation matter.

## 12. References
- Thung, G. & Yang, M. (2016). Classification of Trash for Recyclability Status. Stanford CS229 project.
- TensorFlow/Keras documentation: https://www.tensorflow.org/
- Streamlit documentation: https://docs.streamlit.io/
