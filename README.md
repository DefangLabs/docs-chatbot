# Scikit RAG + OpenAI

This sample demonstrates how to deploy a Flask-based Retrieval-Augmented Generation (RAG) chatbot using OpenAI's GPT model. The chatbot retrieves relevant documents from a knowledge base using scikit-learn and Sentence Transformers and then generates responses using OpenAI's GPT model.

## Prerequisites

1. Download [Defang CLI](https://github.com/DefangLabs/defang)
2. (Optional) If you are using [Defang BYOC](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) authenticated with your AWS account
3. (Optional - for local development) [Docker CLI](https://docs.docker.com/engine/install/)

## Deploying

1. Open the terminal and type `defang login`
2. Type `defang compose up` in the CLI.
3. Your app will be running within a few minutes.

## Local Development

1. Clone the repository.
2. Create a `.env` file in the root directory and set your OpenAI API key or add the OPENAI_API_KEY into your .zshrc or .bashrc file:
3. Run the command `docker compose -f compose.dev.yaml up --build` to spin up a docker container for this RAG chatbot

## Configuration

- The knowledge base is the all the markdown files in the defang docs [website](https://docs.defang.io/docs/intro). The logic for parsing can be found in './app/get_knowledge_base.py'.
- The file `get_knowledge_base.py` parses every webpage as specified into paragraphs and writes to `knowledge_base.json` for the RAG retrieval.
- To obtain your own knowledge base, please feel free to implement your own parsing scheme.
- for local development, please use the compose.dev.yaml file where as for production, please use the compose.yaml.

---

Title: Scikit RAG + OpenAI

Description: An application demonstrating a GPT-4-based chatbot enhanced with a Retrieval-Augmented Generation (RAG) framework, leveraging scikit-learn for efficient contextual embeddings and dynamic knowledge retrieval.

Tags: Flask, Scikit, Python, RAG, OpenAI, GPT, Machine Learning

Languages: python
