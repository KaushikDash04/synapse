# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container to /app
WORKDIR /app

# Copy only the requirements file initially to leverage Docker layer caching
COPY requirements.txt /app/

# Install virtualenv
RUN pip install virtualenv

# Create a virtual environment and activate it
RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 80 available to the world outside this container
EXPOSE 80

# Install python-dotenv to load environment variables from .env file
RUN pip install python-dotenv

# Run the application with Uvicorn ASGI server and load environment variables from .env file
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80", "--env-file", ".env"]
