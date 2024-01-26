# Your final submission name -> E8.zip 

File structure:

- project_report.pdf
- /code
-- test.py
-- weights
-- readme.txt
-- requirements.txt
-- /code/test_data
-- /code/results


## Step-by-Step Guide to Docker
This is a step-by-step guide to creating dokcer image, running the code in a docker container and retrieving results.

#### Create a docker image
1. Clone the repository
2. Open a terminal and navigate to the root directory of the repository
3. Run the following command to create a docker image: `docker build -t <image_name> .`

**_NOTE_** this might take a while

#### Run the docker image
1. Run the following command to list the docker images: `docker image ls`
2. Find the image you just created and copy the image name
3. Run the following command to run the docker image: `docker run <image_name> tail -f`
4. Open a new terminal
5. Run the following command to list the containers and copy the id of the container: `docker container ls`
6. Navigate into the running container: `docker run -it <container_id> bash`

#### Copy additional files into the docker container
1. Open a new terminal
2. Copy checkpoint to the docker container: `docker cp \<path_to_checkpoint>\<checkpoint>.pth <container_id>:/app/checkpoints` 

**_NOTE_** for example: docker cp .\checkpoints\checkpoint_100_100.pth 0f3b6b6b6b6b:/app/checkpoints

#### Execute the code
1. Inside the container navigate to the code directory: `/app`
2. Run the following command to execute the code: `python test_real_integrals.py`

#### Retrieve results
1. Copy the results from the docker container to your local machine: `docker cp <container_id>:/app/visualization_real_integral_0_0.png ./`

**_NOTE_** for example `visualization_real_integral_0_0.png`

#### Stop the docker container
1. Open a new terminal
2. Run the following command to list the running containers: `docker ps`
3. Find the container you just created and copy the container id 
4. Kill the container by running the following command: `docker kill <container_id>`