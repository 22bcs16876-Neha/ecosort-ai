# Viva Questions & Answers

1. **What is machine learning?**  
   A method that allows a computer to learn patterns from data and make predictions.

2. **What is image classification?**  
   Assigning an image to one label from a fixed set of categories.

3. **What is a CNN?**  
   A neural-network architecture designed to learn spatial patterns from images using convolutional filters.

4. **Why do we use MobileNetV2?**  
   It is a comparatively lightweight CNN that is suitable for transfer learning and educational projects.

5. **What is transfer learning?**  
   Reusing a model trained on a large dataset and adapting it to a new task.

6. **Why split the dataset?**  
   Training data teaches the model, validation data helps tune it, and test data estimates performance on unseen examples.

7. **What is data augmentation?**  
   Applying label-preserving transformations such as flips or small rotations to make training more robust.

8. **What is an epoch?**  
   One complete pass through the training dataset.

9. **What is overfitting?**  
   When a model learns training examples too closely and performs poorly on unseen data.

10. **What is a confusion matrix?**  
    A table showing correct and incorrect predictions for each class.

11. **What is precision?**  
    Of the examples predicted as a class, the fraction that truly belongs to that class.

12. **What is recall?**  
    Of all real examples in a class, the fraction correctly identified.

13. **Why is the test set not used during training?**  
    To keep the final evaluation independent and reduce optimistic performance estimates.

14. **What does softmax do?**  
    Converts model output scores into values that sum to 1 and can be interpreted as class probabilities.

15. **Can this model identify every kind of waste?**  
    No. It only predicts among its six training labels and can be uncertain on unfamiliar or mixed items.

16. **Why might the model fail on a real dustbin photo?**  
    The training images may have plain backgrounds and single objects; real scenes contain clutter, occlusion and different lighting.

17. **What is the difference between classification and object detection?**  
    Classification assigns a label to an image; object detection locates and labels multiple objects with bounding boxes.

18. **How can you improve the project?**  
    Add more diverse labelled data, improve class balance, evaluate on real campus images and explore object detection.
