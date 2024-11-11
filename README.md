# Team Info
**Regression Renegades**
* Conner Bell (chb263)
* Kacy Crothers (klc990)
* Joseph Sorrell (jrs1455)
# Instructions

## How to Download Weights
Download the final model weights [here](https://drive.google.com/drive/folders/1OaEbp94YxmSfembZwslnCTwhu-08Klnk?usp=sharing)

## How to Run
Clone the repository then install the requirements
```
pip install -r requirements.txt
```


Download weights and put the weights into the same directoy as the test script. Then run the python test script with the arguments of --model_path and --test_dir

```
--model_path # The directory to where the model weights are
--test_dir # The directory where the test dataset is 
```

Example
```
python directoryTest.py --model_path best_model_11_8_10_13.pth --test_dir TestDataset   
```


# Model Information
## Results Achieved
The following are the final metrics for the latest model:
```
Accuracy: 0.9392
Precision: 0.9402
Recall: 0.9392
F1 Score: 0.9388
```
<img src="https://github.com/user-attachments/assets/e98d3114-6e66-4576-9a20-2cea54164466" height="450">

## Architecture Description
### Tech Stack
* The project was completed using a Jupyter Notebook running v7.2 with Python 3.14
* PyTorch 2.5.1 was heavily used to handle training and TorchVision 0.20.1 was used for image processing and augmentation
* This stack was chosen due to its developer friendly nature and built in machine learning modules

### Image Processing
* The images are pulled from the Google Drive and labeled according to the folder they originated from
* All of the images are randomly split into training, validation, and testing batches at a 7:2:1 ratio
* The training data is tripled by adding two augmented version of each image
* The image augmentations include random a change in orientation, cropping, rotation, brightness, contrast, saturation, and hue
<img src="https://github.com/user-attachments/assets/8e5a6d22-e10c-413a-a7a5-03a030437f94" height="200" width="200" >
<img src="https://github.com/user-attachments/assets/0b457def-5ef9-4276-b825-bbc644e4972a" align="left" height="200" width="200" >
<img src="https://github.com/user-attachments/assets/46729b1b-5704-4669-b5a4-52492c035143" align="left" height="200" width="200" >


### Model iteration count
* The model ran through a total of 9 epochs on a training set of 20920 images
* More information can be found in the model training process section

### Result reporting
* The best model was kept and its accuracy and scores were rerported along with the testing set confusion matrix

## Model Training Process
* Differences between every epoch

# Team Software Process
## Methods for Improvement
* Larger data set
* Photo capture variability
* Data augmentation
### Challenges Encountered
* Swalm vs Lee
* Small Data Size
## Unique Insights
* Comparisons of smaller run times versus larger
* Currently limited by computational power
