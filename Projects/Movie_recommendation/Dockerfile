# Step 1: Base image
FROM python:3.13-slim

# Step 2: Working directory inside container
WORKDIR /app

# Step 3: Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Copy your source code
COPY . .

# Step 5: Default command (runs demo)
CMD ["python", "recommender.py"]
