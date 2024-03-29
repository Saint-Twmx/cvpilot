
# TAVIgator
The code mainly uses the nnUNet framework
https://github.com/MIC-DKFZ/nnUNet

We have conducted some study and validation on Aortic Valve based on the nnUNet code framework.
A total of 183 retrospective patients subjected to CTA of the aorta were included in this study. The mean age of the patients was 69.3 ± 8.4 years; 107 patients (58.5%) were male and 76 patients (41.5%) were female.
The obtained training result metrics are as follows:

![image](https://github.com/Saint-Twmx/TAVIgator/assets/165255758/94b9a363-b215-44c6-93f1-49ec0ff6352e)
![image](https://github.com/Saint-Twmx/TAVIgator/assets/165255758/7c8ea41a-d610-473b-b54a-44ed5c425412)
![image](https://github.com/Saint-Twmx/TAVIgator/assets/165255758/823cdccb-f569-40e8-b8a5-2431ee9e76f7)
![image](https://github.com/Saint-Twmx/TAVIgator/assets/165255758/3ce2fed8-27a2-47b1-b511-1cb2b720a44e)





We present a simplified aortic valve segmentation model trained with a small amount of test data.

# highlight


- We combine machine learning, neural network learning, and traditional algorithms to achieve innovation in solving two-valve and three-valve label mixing and boundary blurring problems, such as aortic valve segmentation.

- We offer a simple solution for dealing with Aortic Valve's quadratic segmentation, as well as a more efficient but more complex solution framework to provide mutual learning.


# Using TAVIgator

- First of all, the test environment of the code is based on python3.9.  You can then install some dependencies such as pytorch with pip install -r requirements.txt. 

- Next, you can partition the model directly by running the "nnUNet_inference.py" file, and using the model we provided.

- After installation, you can activate the training test data we prepared by using the step "python3 plan_and_preprocess.py -t 100". 

- Next you can run the training code by executing the step "python39 run training.py 3d_fullres nnUNetTrainerV2 100 all".

- Note that you need to build the corresponding folder directories in nnUNet's format to execute the next two steps of training code.
  
    -data
  
    --nnUNet_processed
  
    --nnUNet_raw_data_base
  
    ---nnUNet_cropped_data
  
    ---nnUNet_raw_data
  
    ----Task100
  
    --nnUNet_result
  

# How to segment your dats

- You need to put your CT data in the "./TAVIgator/input "directory, it can be an nrrd、nii.gz file,also it can be a dicom folder.

- Next you need to modify the file_name in "nnUNet_inference.py" to specify the name of the file you want to splitl.

- Run "nnUNet_inference.py" and you will get a partition file with the same name in the "./TAVIgator/output "directory.
