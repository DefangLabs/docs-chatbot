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

// Global Variables:
let currentIndex = 0; // Current page index of the bot's response
let chunks; // Message chunks (used when the response exceeds the character limit)
let dotInterval = null;

// Create an express app
const app = express();
// Get port, or default to 3000
const PORT = process.env.PORT || 3000;

// Add health check endpoint
app.get('/', (req, res) => {
  res.status(200).send('OK');
});

// Helper functions below
function startLoadingDots(endpoint, initialMessage) {
  let dotCount = 0;
  let maxDots = 4

  dotInterval = setInterval(() => {
    dotCount = (dotCount % maxDots) + 1;
    const loadingMessage = `${initialMessage}${'.'.repeat(dotCount)}`;
    const options = {
      content: loadingMessage,
      flags: InteractionResponseFlags.EPHEMERAL,
      components: [],
    };

    sendResponse(endpoint, options);
  }, 500); // Interval delay
}

function stopLoadingDots() {
  if (dotInterval) {
    clearInterval(dotInterval);
  }
}

function createMessageWithButtons(index, chunks) {
  currentIndex = index; // Set the global currentIndex to the current index
  return {
    content: chunks[index],
    components: [
      {
        type: 1,
        components: [
          {
            type: 2,
            label: 'Previous',
            style: 1,
            custom_id: `prev_${index}`,
            disabled: index === 0, // Disable if on the first chunk
          },
          {
            type: 2,
            label: 'Next',
            style: 1,
            custom_id: `next_${index}`,
            disabled: index === chunks.length - 1, // Disable if on the last chunk
          },
        ],
      },
    ],
  };
}

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

async function sendResponse(endpoint, options) {
  try {
    const response = await fetch(`https://discord.com/api/v10/${endpoint}`, {
      method: 'PATCH',
      headers: {
        'Authorization': `Bot ${process.env.DISCORD_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        ...options
      }),
    });

    if (!response.ok) {
      console.error(`Failed to send follow-up response. Status: ${response.status}, StatusText: ${response.statusText}`);
    }
  } catch (error) {
    console.error('Error sending follow-up response:', error);
  }
}

async function sendFollowUpResponse(endpoint, followUpMessage) {
  // Check if the follow-up message exceeds Discord's character limit (2000 characters)
  if (followUpMessage.length > 2000) {
    // Split response into chunks of 2000 characters
    chunks = followUpMessage.match(/(.|[\r\n]){1,1990}(?=\s|$)/g) || [];
    // Send the first chunk with prev/next buttons
    await sendResponse(endpoint, createMessageWithButtons(0, chunks));
  } else {
    let options = {
      content: followUpMessage,
      flags: InteractionResponseFlags.EPHEMERAL,
      components: [],
    };
    await sendResponse(endpoint, options);
  }
}

async function fetchFollowUpMessage(question, userId, endpoint) {
  try {
    // Call an external API to fetch the answer
    const answer = await fetchAnswer(question);
    return `\n> ${question}\n\nHere's what I found, <@${userId}>:\n\n${answer}`;
  } catch (error) {
    console.error('Error fetching answer:', error);
    return `\n> ${question}\n\nSorry <@${userId}>, I couldn't fetch an answer to your question. Please try again later.`;
  }
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
      let followUpMessage = "Something went wrong! Please try again later.";

      // Send a placeholder response
      await sendPlaceholderResponse(res, initialMessage);

      // Begin loading dots while fetching follow-up message
      try {
        startLoadingDots(endpoint, initialMessage)
        followUpMessage = await fetchFollowUpMessage(question, userId, endpoint);
      } finally {
        stopLoadingDots()
      }

      // Send the follow-up response
      sendFollowUpResponse(endpoint, followUpMessage);
      
      return;
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

  // Handle button interactions
  if (type === InteractionType.MESSAGE_COMPONENT) {
    const customId = data.custom_id;

    if (customId.startsWith('prev_') || customId.startsWith('next_')) {
      const [action, index] = customId.split('_');
      currentIndex = parseInt(index, 10);

      if (action === 'prev' && currentIndex > 0) {
        currentIndex -= 1;
      } else if (action === 'next' && currentIndex < chunks.length - 1) {
        currentIndex += 1;
      }

      // Respond with the updated message chunk
      return res.send({
        type: InteractionResponseType.UPDATE_MESSAGE,
        data: createMessageWithButtons(currentIndex, chunks),
      });
    }
  }

  console.error('unknown interaction type', type);
  return res.status(400).json({ error: 'unknown interaction type' });
});

app.listen(PORT, () => {
  console.log('Listening on port', PORT);
});
