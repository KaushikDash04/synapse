# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Copy the .env file into the container
COPY .env /app/.env

# Expose port 80 to the outside world
EXPOSE 80

# Command to run the application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--env-file", ".env"]
