## Classifying a major crop in different categories.
Using Machine Learning Algorithm to identify different crop categories in smallholders farms in Africa

![image](https://github.com/MugoDom/crop_damage_classification/assets/132938391/6192f2a2-9227-4f90-8c8a-52d056bb114f)
## Introduction
This post describes how we used the machine learning algorithm to classify crops majorly maize into different categories;Good growth, Nutrient Deficiency, Weed, Drought and Others(Pests, Diseases and fire damage) for easier retrieval and follow up by the insurance to enhance easier payout to farmers incase of any claims. I used this pipeline to enter Zindi’s [Crop Damage Classification](https://github.com/MugoDom/crop_damage_classification/edit/main/README.md#:~:text=Crop%20Damage%20Classification). We may not have won the contest but we learnt some great techniques for working with image classification which I detail in this post.
Here are the preprocessing steps we followed:
1.
2.
3.
4.
5.
6.
7.
8.
The python notebooks I created can be found in this github repository:https:https://github.com/MugoDom/crop_damage_classification/blob/main/index.ipynb
## The Challenge
Zindi is an African competitive data science platform that focusses on using data science for social benefit. In Zindi’s 2019 Farm Pin Crop Detection Challenge, participants to **trained machine learning models using Image Classification  in order to classify the crops being grown in fields in Africa.

The data used was gathered from the pictures sent by insured farmers from their smartphone.The data supplied to contestants consisted of two shape files containing the training set and test set.
![image](https://github.com/MugoDom/crop_damage_classification/assets/132938391/73dad8af-78be-44d7-b600-5beeb47b1e38)
The bar graph above shows the damage plot distribution analysis.
The training and test sets consisted of 26068 fields and 8663 fields respectively. Each field in the training set was labelled with the ID, damage and file name. The crop type was majorly Maize and it was classified into different categories;
1. Good growth (G)
2. Drought (DR)
3. Nutrient Deficient (ND)
4. Weed (WD)
5. Other (including pest, disease, or wind damage)
## Data preprocessing.
### Dividing the Dataset into Training and Testing Sets
we chose to partition the provided training data into distinct train and test sets. This decision was made to guarantee the availability of a dedicated test set for evaluating our optimized model.
### Organizing the images
We first defined the source and destination directories for the images, created the destination directories if they didnt exist and moved the images to their respective directories.Went ahead and checked the number of images in each directory to confirm the data balancing. 
 We did the minority class oversampling to handle class imbalance issues.
 ### Memory-Efficient Data Loading
 Given the substantial size of our dataset, employing traditional preprocessing techniques—such as loading the entire dataset into memory—would have posed significant memory challenges. In light of this, we opted for the use of the ImageDataGenerator.
 ## Building a prediction model
 The organized training images in the base directory were resized to (224,224) pixels processing them in batches of 256 during training.
 ### Data Augmentation
 1. rescaling
 2. horizontal flips
 3. zoom
 4. shear
we used the above to enhance the models efficiency.
## Training and Validation
The data was split into training and validation sets using validation_split=0.2.
We visualized the training and validation performance metrics using two subplots; T
a) The loss; 
b) The Accuracy plot as shown below
![image](https://github.com/MugoDom/crop_damage_classification/assets/132938391/5c689812-8157-4218-a130-aae547e9acfd)
Confusion Matrix for the base model 
![image](https://github.com/MugoDom/crop_damage_classification/assets/132938391/cb981366-7ec4-449a-956c-499c553ba3ab)
### Optimized Model Evaluation
![image](https://github.com/MugoDom/crop_damage_classification/assets/132938391/6afb43f7-652b-4c0b-88a7-b31dd77079cb)
### Results and Areas of Improvement
The imbalance in the classes was one of the biggest problems we encountered. This clearly had an impact on the model. We attempted to improve the model by oversampling the minority class, but the results don't seem to be very noticeable. The low F1 score could be partially explained by this. We also tried with adding class weighting to the model, but this did not improve performance; instead, it increased overfitting. After giving this some thought, we'll investigate alternative approaches like loss functions, which, unlike cross entropy, consider every label equally and so account for the class imbalance in the dataset.

We also plan to investigate the fastai deep learning library, which offers high-level building blocks that quickly yield "state-of-the-art" outcomes. 
Another challenge was running out of compute resources limiting the number of epochs we can run. Given enough computing resources we could run more epochs with guaranteed improvement to performance.



 
 
