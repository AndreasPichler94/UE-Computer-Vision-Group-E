# Team E8 Readme 


# Quick start
2 modes of running the test.py and train.py scripts are provided:
1) anaconda virtual environment 
2) docker container

If there is an issue with running the scripts locally in an anaconda virtual environment we recommend the approach using a docker container.
Once the environment is set up correctly please move on to the "Execute the code" section

## Step-by-Step Guide to Anaconda

#### Create a virtual environment
1. Open a terminal and navigate to the "code" directory
2. Run the following command to create a new virtual environment: `conda create -n "aos_e8" python=3.11.7`
3. Activate the environment by running: `conda activate aos_e8`
4. Install dependencies by running: `pip install -r requirements.txt`


## Step-by-Step Guide to Docker
This is a step-by-step guide to creating dokcer image, running the code in a docker container and retrieving results.

#### Create a docker image
1. Open a terminal and navigate to the "code" directory
2. Run the following command to create a docker image: `docker build -t aos_e8 .`


### Required files
1. The necessary model weights can be downloaded from: https://www.dropbox.com/scl/fi/b42egymxrsse1mgcm4z53/weights.zip?rlkey=45i4ouaszta5y6dabyzmyeqjk&dl=0
2. The test data (not seen by the model before) can be downloaded from: https://www.dropbox.com/scl/fi/xsrz91uc4wawso7cgtnvq/test-data-folder.zip?rlkey=a57jh9yml63gxk9xx7jbt40m0&dl=0
3. The model weights should be put inside the "weights" folder, the test data inside the "test data folder" folder.
4. After you have downloaded the necessary filed, copy them into the container:
`docker cp "weights" "aos_e8:/code"`
`docker cp "test data folder" "aos_e8:/code"`

#### Run the docker image
1. Run the following command to run the docker image from the "code" directory: 
`docker run -it -v ./:/code --name aos_e8 --gpus all aos_e8`  
2. On some Windows machines, you may need to modify this command:
`docker run -it -v "C:\Users/yourUsername/path/to/code/:/code --name aos_e8 --gpus all aos_e8`
3. If this doesn't work, try it without the GPU flags (Note: Code may not work properly without GPU):
`docker run -it -v "C:\Users/yourUsername/path/to/code/:/code --name aos_e8 aos_e8`
4. Run the following command to execute the code for testing on test data that the model has never seen: `python test.py`. As a comparison the outputs that we generated can be found under: 
5. Run the following command to execute the code for testing on real integrals: `python test_real_integrals.py`
6. Run the following command to train a new model: `python train.py`. The train data should be in the same directory under `./train`, test data under `test`.


# Retrieve results
1. Outputs for the test set can be found under "Result folder", outputs for the real integrals can be found in the code folder, the file name is "visualization_real_integral_0_0.png"
