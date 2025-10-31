# Use the official PyTorch image which already includes PyTorch, CUDA, cuDNN, and Python
# The 'runtime' tag is generally smaller than 'devel' and is suitable for running applications.
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# Set environment variable to prevent Python from buffering its output
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# The necessary build tools, git, wget, curl, ca-certificates, and display libraries 
# are typically included or not needed in a PyTorch-based image.
# If you find you need any of these, you can add them back, but they are often
# part of the base layer.
# We'll skip most apt-get calls as the base image is already set up.

WORKDIR /app

# Copy the requirements file first to take advantage of Docker layer caching
COPY requirements.txt /app/

# Install the dependencies from requirements.txt
# This ensures PyTorch dependencies are correctly handled by the base image
# Remove the incorrect 'tensorflow-gpu' install unless your project specifically uses BOTH frameworks.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . /app

# Set the entry point or command if needed, but 'tty: true' and 'stdin_open: true'
# in docker-compose are often used for interactive sessions.
# CMD ["python", "your_script.py"]