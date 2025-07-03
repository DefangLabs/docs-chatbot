#!/bin/bash

set -e

# Define the path for the defang-docs directory
DEFANG_DOCS_PATH=$(readlink -f ./defang-docs)

# Install npm packages in defang-docs
npm -C $DEFANG_DOCS_PATH install
npm -C $DEFANG_DOCS_PATH run prebuild
