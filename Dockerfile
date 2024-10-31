# Use an official Python runtime as a parent image
#FROM python:3.10-slim-buster
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements file into the container
COPY ./requirements.txt /app/requirements.txt

# Install gcc and other dependencies
RUN apt-get update && apt-get install -y \
    gcc && \
    rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

