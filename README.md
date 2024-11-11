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
--model_path # The directory to where the model weights are (Use "best_model_11_8_10_13.pth" for the latest model)
--test_dir # The directory where the test dataset is 
```

Example
```
python directoryTest.py --model_path best_model_11_8_10_13.pth --test_dir TestDataset   
```
**Note: best_model_11_8_10_13.pth is the latest model that is being submitted. However, other models are present from past runs and for comparison purposes.**

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
* PyTorch 2.5.1 was used to handle training and TorchVision 0.20.1 was used for image processing and augmentation
* A pre-trained model, Resnet-18, was used as the base model
* The model was trained using a GForce RTX 4070
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
* The best model was kept and its accuracy and scores were reported along with the testing set confusion matrix

## Model Training Process
* For every epoch the model continuously approved accross both metrics
<img width="412" src="https://github.com/user-attachments/assets/23bb17e1-9506-410d-9e34-9a084efc5c34">
<img width="412" src="https://github.com/user-attachments/assets/7bfb14c9-c921-431c-a810-8354ebdd341a">
<img width="412" src="https://github.com/user-attachments/assets/fe05fa69-cfaa-4b48-8835-1935a0faf5f8">


# Team Software Process
## Methods for Improvement
### Larger Data Set
* In one of the early models that was run, the data set was around 2000 training images which was much too low to get the desired accuracy
* The metrics of the early model, trained for nine epochs on a small data set, were:
```
Accuracy: 0.8263
Precision: 0.8337
Recall: 0.8263
F1 Score: 0.8270
```
* To solve this problem, the team captured more images to improve the data set to a higher volume that resulted in an eventual 20920 training images after augmentations
* Increasing the data set size was the single biggest improvement in model accuracy, but there were other noteworthy improvements
### Decreasing the Learning Rate
* The learning rate of the original model was set too high
* This led to an overfitting of the data and a potential for a missed minimum val loss
* By changing it to a lower learning rate, we ensured that the model would appropriately approach the limit and not oscillate between val loss values
### Photo Capture Variance
* Photos captured of the buildings were taken at different times of day to ensure different lighting and weather conditions for training; an example of Butler Hall on three different days can be seen below
<img width="200" src="https://github.com/user-attachments/assets/7e9b83a1-faa5-498e-8438-52dfed5c125c">
<img width="200" src="https://github.com/user-attachments/assets/844b0de5-d038-4893-a1b7-a10142f62260">
<img width="200" src="https://github.com/user-attachments/assets/e1fcc950-2c27-4d9c-90be-140dec7af959">

### Image Augmentation
* Images were occasionally being augmented and cropped so heavily that the data set did not meaningfully contribute to the model
<img width="200" src="https://github.com/user-attachments/assets/edfe846f-2f17-4b0a-a643-ec392a36d83b">
<img width="200" src="https://github.com/user-attachments/assets/d9fc229d-9383-4817-87e0-ef9ef84652ec">

* Obviously, these images only served to confuse the model. The amount of the augmentation, specifically cropping, of the training set was decreased and the model showed improvement

## Challenges Encountered
### Swalm vs Lee
* An interesting challenge that appeared was that in an early model the images of Swalm Hall and Lee Hall were easily mislabeled. This is understandable due to the buildings being nearly identical apart from their surroundings.
<img width="300" src="https://github.com/user-attachments/assets/56aa9426-4e17-41ff-8299-3ccf8f528f40">

* The problem was solved by adding wider images of the buildings to the data set to help the model process its surrounding. A larger volume of images in the training set probably contributed to solving the problem as well.

* An alternative solution that was considered but eventually not implemented was the creation of a very selective second layer of a model that focused only on labeling images between Swalm and Lee.
  
### Bad Photos

* Another challenge was taking pictures that were quality and distinguishable for the model. Many images of buildings ended up simply being pictures of trees or passing vehicles on a street which no model could be expected to accurately predict. Two particularly bad photos that were removed can be seen below:
<img width="200" src="https://github.com/user-attachments/assets/0c8355f6-d583-40a8-8110-653487e0ecb6">
<img width="200" src="https://github.com/user-attachments/assets/2c16bd8e-b356-454b-b416-ccd60aea7114">

* The team solved this problem by manually browsing through all images in the data set and deleting any that were judged to be indistuigishable by a well trained computer
  
## Unique Insights
### Currently limited by computational power
* The biggest insight we gained from the process was the amount of computational power that is needed to process large sets of images. Currently, the model was trained on a GForce RTX 4070. Nine epochs took the model over seven hours to process and train.
* The val loss and accuracy continued to improve, albeit at slower rates, as the model trained and never flattened off or dropped. This indicates that further epochs could potentially enhance the models accuracy. However, due to the constraints of our computational power, further epochs were impossible to obtain.
* Moving forward, if we were to continue to improve the model, we would look into utilizing the campus super computer to process the model and expand the number of epochs until the val loss stops decreasing. 
