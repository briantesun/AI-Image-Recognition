# AI-Image-Recognition

Summary: A classification model that leverages transfer learning with the VGG16 architecture and pre-trained ImageNet weights. Initially, the top layers remain frozen during training to preserve the learned features. Data augmentation techniques enhance the training dataset, and fine-tuning the entire model improves its performance. Evaluating the model on a validation dataset ensures its accuracy and effectiveness in classifying fresh and rotten fruit.

You can find the dataset I used here: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

Model summary:

![Model summary](https://user-images.githubusercontent.com/129823285/230555613-c08b8819-ed93-4b6d-b427-f15b7b6fbcf0.png)

Training (before fine-tuning):
![Training (before fine-tuning)](https://user-images.githubusercontent.com/129823285/230555638-71952183-b0ae-47ea-b3d5-671c4fd17e62.png)

Training (after fine-tuning):
![Training (after fine-tuning)](https://user-images.githubusercontent.com/129823285/230555652-27ae1e47-3508-4103-ae02-b0c4ed074de5.png)

Results:

![Results](https://user-images.githubusercontent.com/129823285/230555704-19e24d89-0f1e-4265-a1ef-018bc22ab67a.png)
