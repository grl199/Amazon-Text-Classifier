# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Pipfile and Pipfile.lock into the container
COPY Pipfile Pipfile.lock /app/

# Install pipenv and project dependencies
RUN pip install pipenv && pipenv install --deploy --ignore-pipfile

# Download NLTK data
RUN pipenv run python -m nltk.downloader punkt_tab

# Create the model directory
RUN mkdir -p /app/model

# Copy the rest of the application code into the container
COPY src/ /app/src/
COPY img/ /app/img/
COPY scripts/config.yaml /app/scripts/
COPY model/ /app/model/
COPY data/ /app/data/

# Expose the port the app runs on
EXPOSE 4000

# Set the default command to run the application
CMD ["pipenv", "run", "python", "src/main.py", "--command", "${COMMAND}", "--model", "${MODEL}"]
