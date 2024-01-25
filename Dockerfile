# TODO: work in progress
# Use an official Python runtime as a parent image
FROM python:3.12.1-windowsservercore-1809



# Download and install Python 3.7.9
#ADD https://www.python.org/ftp/python/3.7.9/python-3.7.9-amd64.exe C:\python-3.7.9-amd64.exe
#RUN C:\python-3.7.9-amd64.exe /quiet InstallAllUsers=1 PrependPath=1 && \
#    del C:\python-3.7.9-amd64.exe
# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run test.py when the container launches
CMD ["python", "hello.py"]
