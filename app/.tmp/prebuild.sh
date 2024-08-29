#!/bin/bash

set -e

CWD=$(pwd)

# Define the path for the defang-docs directory
DEFANG_DOCS_PATH=$(readlink -f ./defang-docs)
CLI_DOCS_PATH="$DEFANG_DOCS_PATH/docs/cli"

# Define the path for the defang directory and set the command directory path
DEFANG_PATH=$(readlink -f ./defang)
GEN_DOCS_CMD_PATH="$DEFANG_PATH/src/cmd/gendocs"

# Navigate to the gendocs command directory and run the Go command
cd "$GEN_DOCS_CMD_PATH"

# Ensure all dependencies are downloaded
go mod tidy
go mod download

# Run the Go command
go run main.go "$CLI_DOCS_PATH"

# Install npm packages in defang-docs
cd "$DEFANG_DOCS_PATH"
npm install

# Return to original directory and run node script
cd "$CWD"
node "$DEFANG_DOCS_PATH/scripts/prep-cli-docs.js"
