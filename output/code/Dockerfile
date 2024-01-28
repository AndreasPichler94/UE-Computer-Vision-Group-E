# Use an official Python runtime as a parent image
FROM python:3.11.7-bookworm

WORKDIR /app

COPY . ./

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# CMD ["python", "test_real_integrals.py"]
