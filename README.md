# Scikit RAG + OpenAI & Discord App for Defang

This repository contains two projects:

1. **Scikit RAG + OpenAI** in `/app`: A Flask-based Retrieval-Augmented Generation (RAG) chatbot using OpenAI's GPT model, scikit-learn, and Sentence Transformers for dynamic knowledge retrieval.

2. **Discord App for Defang** in `/discord-bot`: A Discord bot designed for Defang Software Labs, providing helpful resources and interacting with users via slash commands.

---

# Scikit RAG + OpenAI

### Overview

This application demonstrates how to deploy a Flask-based Retrieval-Augmented Generation (RAG) chatbot using OpenAI's GPT model. The chatbot retrieves relevant documents from a knowledge base using scikit-learn and Sentence Transformers and then generates responses using OpenAI's GPT model.

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
2. Create a `.env` file in the root directory and set your OpenAI API key, or add the `OPENAI_API_KEY` to your `.zshrc` or `.bashrc` file.
3. Run the command:

   ```bash
   docker compose -f compose.dev.yaml up --build
   ```

   This spins up a Docker container for the RAG chatbot.

## Configuration

- The knowledge base is the all the markdown files in the Defang docs [website](https://docs.defang.io/docs/intro). The logic for parsing can be found in `./app/get_knowledge_base.py`.
- The file `get_knowledge_base.py` parses every webpage as specified into paragraphs and writes to `./data/knowledge_base.json` for the RAG retrieval.
- To obtain your own knowledge base, please feel free to implement your own parsing scheme.
- for local development, please use the `compose.dev.yaml` file where as for production, please use the `compose.yaml`.

---

# Discord App for Defang

### Overview

This is a Discord bot developed for [Defang Software Labs](https://github.com/DefangLabs). It provides helpful resources in a Discord server and interacts with users via slash commands. The bot is built using Discord's official [template](https://github.com/discord/discord-example-app).

## Features

### Slash Commands

`/ask`: A command to ask Defang-related questions to the bot. The bot accesses the Ask Defang (ask.defang.io) API endpoint for retrieving responses.

`/test`: A basic command to test functionality using the Discord API, without relying on external APIs.

## Development

### Project structure

Below is a basic overview of the project structure:

```
├── .env.       -> .env file (not shown)
├── app.js      -> main entrypoint for app
├── commands.js -> slash command payloads + helpers
├── utils.js    -> utility functions and enums
├── package.json
├── README.md
└── .gitignore
```

### Setup project

Before you start, you'll need to install [NodeJS](https://nodejs.org/en/download/) and [create a Discord app](https://discord.com/developers/applications) with the proper permissions:

- `applications.commands`
- `bot` (with Send Messages enabled)
  Configuring the app is covered in detail in the [getting started guide](https://discord.com/developers/docs/getting-started).

### Install dependencies

```
cd discord-bot
npm install
```

### Get app credentials

Fetch the credentials from your app's settings and add them to a `.env` file. You'll need your app ID (`DISCORD_APP_ID`), bot token (`DISCORD_TOKEN`), and public key (`DISCORD_PUBLIC_KEY`).
You will also need an `ASK_TOKEN` to authenticate API calls to the Ask Defang endpoint.

### Install slash commands

The commands for the example app are set up in `commands.js`. All of the commands in the `ALL_COMMANDS` array at the bottom of `commands.js` will be installed when you run the `register` command configured in `package.json`:

```
cd discord-bot
npm run register
```

### Running the app locally

After your credentials are added, go ahead and run the app:

```
cd discord-bot
npm run start
```

### Set up interactivity

The project needs a public endpoint where Discord can send requests. To develop and test locally, you can use something like [`ngrok`](https://ngrok.com/) to tunnel HTTP traffic.

Install ngrok if you haven't already, then start listening on port `3000` in a separate terminal:

```
ngrok http 3000
```

You should see your connection open:

```
Tunnel Status                 online
Version                       2.0/2.0
Web Interface                 http://127.0.0.1:4040
Forwarding                    https://1234-someurl.ngrok.io -> localhost:3000

Connections                  ttl     opn     rt1     rt5     p50     p90
                              0       0       0.00    0.00    0.00    0.00
```

Copy the forwarding address that starts with `https`, in this case `https://1234-someurl.ngrok.io`, then go to your [app's settings](https://discord.com/developers/applications).

On the **General Information** tab, there will be an **Interactions Endpoint URL**. Paste your ngrok address there, and append `/interactions` to it (`https://1234-someurl.ngrok.io/interactions` in the example).

Click **Save Changes**, and your app should be ready to run 🚀
