# Use an appropriate base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# List files in the directory for debugging
RUN ls -l /app

# Expose the application port
EXPOSE 5000

# Set the Flask application environment variable
ENV FLASK_APP=myapp:create_app

# Run the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
