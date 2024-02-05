# Use the official Python 3.11 base image from Microsoft's devcontainers
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

# Update the system and install required packages
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# Set the working directory to /workspaces/person-detection
WORKDIR /

# Copy the contents of the current directory into the container at /workspaces/person-detection
COPY . .

# Run post-create command to install Python dependencies
RUN pip3 install --user -r /requirements.txt

# Expose port 8001 for the FastAPI application
ENV PATH="/root/.local/bin:${PATH}"

# Run the FastAPI application without automatic reload
CMD ["bash", "-c", "PORT=8080 ./run.sh"]
