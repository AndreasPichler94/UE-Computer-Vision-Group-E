# Use an official Python runtime as a parent image
FROM python:3.11.7-bookworm

WORKDIR /requirements

# Copy the current directory contents into the container at /app
COPY ./requirements.txt /requirements/

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /code

CMD ["/bin/bash"]

# Run test.py when the container launches
# CMD ["python", "hello.py"]
#CMD ["python", "test_real_integrals.py"]
