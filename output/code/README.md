# Group E8 - code reproduction

Please note that David stopped interacting with the group and didn't contribute anything, as indicated in workload.csv

## Instructions to run the model and reproduce the real integral

#### Create and run the docker image
1. Clone the repository
2. Open a terminal and navigate to the root directory of the repository
3. Run the following command to create a docker image: `docker build -t <image_name> .` (this will take a few minutes)
4. Launch and shell into the container: `docker run -it <image_name> bash`. Typing into the bash console:
5. Evaluate the model on the real drone image: `python test_real_integrals.py`
6. (Optional: To launch the training script, run `python train.py`. This step requires AOS integral training data to be present in the /app/data/train folder )
7. Open a new terminal and copy the results from the docker container to your local machine: `docker cp <container_id>:/app/visualization_real_integral_0_0.png ./`


#### Stop the docker container
1. Open a new terminal
2. Run the following command to list the running containers: `docker ps`
3. Find the container you just created and copy the container id 
4. Kill the container by running the following command: `docker kill <container_id>`