import dotenv from 'dotenv';
import path from 'path';

// Load environment variables from the parent folder
const __dirname = path.dirname(new URL(import.meta.url).pathname);
dotenv.config({ path: path.resolve(__dirname, '../.env') });

import express from 'express';
import {
  InteractionResponseFlags,
  InteractionResponseType,
  InteractionType,
  verifyKeyMiddleware,
} from 'discord-interactions';
import { getRandomEmoji } from './utils.js';

// Create an express app
const app = express();
// Get port, or default to 3000
const PORT = process.env.PORT || 3000;

// Add health check endpoint
app.get('/', (req, res) => {
  res.status(200).send('OK');
});

// Helper functions below
async function sendPlaceholderResponse(res, placeholderResponse) {
  await res.send({
    type: InteractionResponseType.CHANNEL_MESSAGE_WITH_SOURCE,
    data: {
      content: placeholderResponse,
      flags: InteractionResponseFlags.EPHEMERAL,
      components: [], 
    },
  });
}

async function fetchAnswer(question) {
  const response = await fetch('https://ask.defang.io/v1/ask', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${process.env.ASK_TOKEN}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query: question }),
  });

  const rawResponse = await response.text();
  console.log('Raw API response:', rawResponse);

  if (!response.ok) {
    throw new Error(`API error! Status: ${response.status}`);
  }

  return rawResponse || 'No answer provided.';
}

async function sendFollowUpResponse(endpoint, content) {
  await fetch(`https://discord.com/api/v10/${endpoint}`, {
    method: 'PATCH',
    headers: {
      'Authorization': `Bot ${process.env.DISCORD_TOKEN}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      content,
      flags: InteractionResponseFlags.EPHEMERAL,
      components: [],
    }),
  });
}

/**
 * Interactions endpoint URL where Discord will send HTTP requests
 * Parse request body and verifies incoming requests using discord-interactions package
 */
app.post('/interactions', verifyKeyMiddleware(process.env.DISCORD_PUBLIC_KEY), async function (req, res) {
  // Interaction id, type and data
  const { id, type, data } = req.body;

  /**
   * Handle verification requests
   */
  if (type === InteractionType.PING) {
    return res.send({ type: InteractionResponseType.PONG });
  }

  /**
   * Handle slash command requests
   * See https://discord.com/developers/docs/interactions/application-commands#slash-commands
   */
  if (type === InteractionType.APPLICATION_COMMAND) {
    const { name } = data;

    // "ask command"
    if (name === 'ask') {
      const context = req.body.context;
      const userId = context === 0 ? req.body.member.user.id : req.body.user.id
      
      const question = data.options[0]?.value || 'No question provided';
      const endpoint = `webhooks/${process.env.DISCORD_APP_ID}/${req.body.token}/messages/@original`;
      const initialMessage = `\n> ${question}\n\nLet me find the answer for you. This might take a moment`

      // Send a placeholder response
      await sendPlaceholderResponse(res, initialMessage);

      // Show animated dots in the message while waiting
      let dotCount = 0;
      const maxDots = 4;
      let isFetching = true;

      const interval = setInterval(() => {
        if (isFetching) {
          dotCount = (dotCount % maxDots) + 1;
          sendFollowUpResponse(endpoint, `${initialMessage}${'.'.repeat(dotCount)}`);
        }
      }, 500);

      // Create the follow-up response
      let followUpMessage;
      try {
        // Call an external API to fetch the answer
        const answer = await fetchAnswer(question);
        followUpMessage = `\n> ${question}\n\nHere's what I found, <@${userId}>:\n\n${answer}`;
      } catch (error) {
        console.error('Error fetching answer:', error);
        followUpMessage = `\n> ${question}\n\nSorry <@${userId}>, I couldn't fetch an answer to your question. Please try again later.`;
      } finally {
        // Ensure cleanup and state updates
        isFetching = false; // Mark fetching as complete
        clearInterval(interval); // Stop the dot interval
      }

      return sendFollowUpResponse(endpoint, followUpMessage);
  }
  
    // "test" command
    if (name === 'test') {
      // Send a message into the channel where command was triggered from
      return res.send({
        type: InteractionResponseType.CHANNEL_MESSAGE_WITH_SOURCE,
        data: {
          // Fetches a random emoji to send from a helper function
          content: `Develop Anything, Deploy Anywhere ${getRandomEmoji()}`,
        },
      });
    }

    console.error(`unknown command: ${name}`);
    return res.status(400).json({ error: 'unknown command' });
  }

  console.error('unknown interaction type', type);
  return res.status(400).json({ error: 'unknown interaction type' });
});

app.listen(PORT, () => {
  console.log('Listening on port', PORT);
});
