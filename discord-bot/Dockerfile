# Use the slim version of Node.js on Debian Bookworm as the base image
FROM node:20-bookworm-slim

RUN apt-get update -qq \
    && apt-get install -y curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json to the container
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code to the container
COPY . .

# Expose the port the app runs on (adjust if necessary)
EXPOSE 3000

# Define the command to run the application
CMD ["npm", "run", "start"]