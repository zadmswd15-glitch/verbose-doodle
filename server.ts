import express from 'express';
import path from 'path';
import dotenv from 'dotenv';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const app = express();
const PORT = 3000;

// Initialize GoogleGenAI SDK with required telemetry User-Agent
const apiKey = process.env.GEMINI_API_KEY;
const ai = new GoogleGenAI({
  apiKey: apiKey,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    }
  }
});

// Helper to check if an API key is a default placeholder or empty string
const isPlaceholderKey = (key: string): boolean => {
  if (!key) return true;
  const k = key.trim();
  return (
    k.includes('gsk_sjlyhcUQy3il') ||
    k.includes('AQ.Ab8RN6LUUD') ||
    k.includes('sk-svcacct-jYJe7HbO6k') ||
    k === 'gsk_...' ||
    k === 'sk-...' ||
    k === 'AIzaSy...'
  );
};

// Middleware for parsing JSON and large request payloads
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// Server API Routes
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', time: new Date().toISOString() });
});

// Advanced Chat API Endpoint
app.post('/api/chat', async (req, res) => {
  try {
    const { 
      message, 
      history = [], 
      systemInstruction = '', 
      temperature = 0.7, 
      attachments = [], 
      learnedKnowledge = [],
      activeProvider = 'gemini',
      openaiKey = '',
      groqKey = '',
      geminiKey = '',
      nanabananaKey = '',
      openaiModel = 'gpt-4o-mini',
      groqModel = 'llama-3.3-70b-versatile',
      geminiModel = 'gemini-3.5-flash'
    } = req.body;

    if (!message && attachments.length === 0) {
      return res.status(400).json({ error: 'Message or attachments are required' });
    }

    // Synthesize grounding knowledge into system instruction to achieve "long-term memory"
    let finalSystemInstruction = "You are a professional, friendly, and highly intelligent bilingual AI assistant. You answer elegantly in both Arabic and English depending on the language of the prompt.\n\n";
    
    if (learnedKnowledge && learnedKnowledge.length > 0) {
      finalSystemInstruction += "### WHAT YOU HAVE LEARNED ABOUT THE USER (LONG-TERM MEMORY):\n";
      learnedKnowledge.forEach((k: any) => {
        finalSystemInstruction += `- ${k.fact}\n`;
      });
      finalSystemInstruction += "\nUse this learned knowledge contextually to personalize and enhance your responses when relevant, but do not mention that you are pulling it from memory unless asked.\n\n";
    }

    if (systemInstruction) {
      finalSystemInstruction += `### ADDITIONAL CUSTOM PERSONALITY / INSTRUCTIONS:\n${systemInstruction}\n`;
    }

    // ----------------------------------------------------
    // Provider 1: OpenAI GPT Connection
    // ----------------------------------------------------
    if (activeProvider === 'openai') {
      const activeOpenaiKey = (!openaiKey || isPlaceholderKey(openaiKey)) ? process.env.OPENAI_API_KEY : openaiKey;
      if (!activeOpenaiKey) {
        return res.status(400).json({ 
          error: 'OpenAI API key is not connected. Please connect it in the Workspace & API Hub (Connections sidebar tab) to use GPT-4o.' 
        });
      }

      const messages = [];
      if (finalSystemInstruction) {
        messages.push({ role: 'system', content: finalSystemInstruction });
      }
      history.forEach((h: any) => {
        messages.push({ role: h.role === 'model' ? 'assistant' : 'user', content: h.content });
      });

      let userContent = message || '';
      if (attachments.length > 0) {
        userContent += `\n\n[Attached Files: ${attachments.map((a: any) => a.name).join(', ')}]`;
      }
      messages.push({ role: 'user', content: userContent });

      const openAiRes = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${activeOpenaiKey}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: openaiModel || 'gpt-4o-mini',
          messages,
          temperature
        })
      });

      if (!openAiRes.ok) {
        const errText = await openAiRes.text();
        throw new Error(`OpenAI API Error: ${errText}`);
      }

      const openAiData = await openAiRes.json();
      return res.json({ 
        text: openAiData.choices?.[0]?.message?.content || 'No response from OpenAI GPT.',
        timestamp: Date.now()
      });
    }

    // ----------------------------------------------------
    // Provider 2: Groq Connection
    // ----------------------------------------------------
    if (activeProvider === 'groq') {
      const activeGroqKey = (!groqKey || isPlaceholderKey(groqKey)) ? process.env.GROQ_API_KEY : groqKey;
      if (!activeGroqKey) {
        return res.status(400).json({ 
          error: 'Groq API key is not connected. Please connect it in the Workspace & API Hub to use ultra-fast LLaMA-3.' 
        });
      }

      const messages = [];
      if (finalSystemInstruction) {
        messages.push({ role: 'system', content: finalSystemInstruction });
      }
      history.forEach((h: any) => {
        messages.push({ role: h.role === 'model' ? 'assistant' : 'user', content: h.content });
      });

      let userContent = message || '';
      if (attachments.length > 0) {
        userContent += `\n\n[Attached Files: ${attachments.map((a: any) => a.name).join(', ')}]`;
      }
      messages.push({ role: 'user', content: userContent });

      // Try the requested model first, fallback to llama-3.3-70b-versatile/llama-3.1-8b-instant if not available
      const requestedGroqModel = groqModel || 'llama-3.3-70b-versatile';
      const modelsToTry = Array.from(new Set([
        requestedGroqModel,
        'llama-3.3-70b-versatile',
        'llama-3.1-8b-instant',
        'mixtral-8x7b-32768'
      ]));
      let groqRes;
      let lastError = '';

      for (const groqModel of modelsToTry) {
        try {
          groqRes = await fetch('https://api.groq.com/openai/v1/chat/completions', {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${activeGroqKey}`,
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              model: groqModel,
              messages,
              temperature
            })
          });

          if (groqRes.ok) {
            break;
          } else {
            const errText = await groqRes.text();
            lastError = `Model ${groqModel} failed: ${errText}`;
            console.warn(lastError);
          }
        } catch (e: any) {
          lastError = `Fetch for ${groqModel} failed: ${e.message}`;
          console.warn(lastError);
        }
      }

      if (!groqRes || !groqRes.ok) {
        throw new Error(`Groq API Error: ${lastError || 'Failed all models'}`);
      }

      const groqData = await groqRes.json();
      return res.json({ 
        text: groqData.choices?.[0]?.message?.content || 'No response from Groq.',
        timestamp: Date.now()
      });
    }

    // ----------------------------------------------------
    // Provider 3: Nana Banana Whimsical API
    // ----------------------------------------------------
    let customGeminiKey = geminiKey;
    if (activeProvider === 'nanabanana') {
      finalSystemInstruction += `\n\n### NANA BANANA PROTOCOL ENGAGED:
You are Nana Banana AI, a whimsical banana assistant! Keep your answers incredibly joyful, fun, and yellow!
1. You must answer utilizing funny banana terms, jokes, or comparisons (e.g., "That is totally bananas!", "This is so a-peeling!", "Let's peel this down!").
2. Connect everything to bananas, monkeys, jungle vines, or smoothies.
3. Keep the user smiling with delightful humor and banana metaphors in both English and Arabic!
4. If the user provided a custom Nana Banana Key (${nanabananaKey ? 'Key Present: ' + nanabananaKey.substring(0, 4) + '...' : 'No Key'}), acknowledge it creatively as your "Golden Banana Permit"!`;
    }

    // ----------------------------------------------------
    // Provider 4: Gemini / Google AI Studio Connection (Default)
    // ----------------------------------------------------
    const activeGeminiKey = (!customGeminiKey || isPlaceholderKey(customGeminiKey)) ? apiKey : customGeminiKey;
    if (!activeGeminiKey) {
      return res.status(500).json({ 
        error: 'GEMINI_API_KEY is not configured. Please configure your API key or connect it in Settings.' 
      });
    }

    // Initialize specific Gemini SDK instance
    const hasCustomGeminiKey = customGeminiKey && !isPlaceholderKey(customGeminiKey);
    const activeAi = hasCustomGeminiKey ? new GoogleGenAI({
      apiKey: customGeminiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        }
      }
    }) : ai;

    // Build parts for current user input
    const currentParts: any[] = [];

    // Parse attachments and append them as inline parts
    if (attachments && attachments.length > 0) {
      for (const att of attachments) {
        let mimeType = att.type || '';
        if (!mimeType) {
          const ext = att.name.split('.').pop()?.toLowerCase();
          if (ext) {
            const textExtensions = ['txt', 'md', 'py', 'js', 'ts', 'tsx', 'jsx', 'html', 'css', 'json', 'csv', 'log', 'yaml', 'yml', 'ini', 'conf', 'sql', 'sh', 'bash'];
            if (textExtensions.includes(ext)) {
              mimeType = 'text/plain';
            } else if (ext === 'pdf') {
              mimeType = 'application/pdf';
            } else if (['png', 'jpg', 'jpeg', 'webp', 'gif'].includes(ext)) {
              mimeType = `image/${ext === 'jpg' ? 'jpeg' : ext}`;
            } else {
              mimeType = 'application/octet-stream';
            }
          } else {
            mimeType = 'application/octet-stream';
          }
        }

        // Standardize any non-standard or custom text/code/binary types to Gemini-supported MIME types
        const knownMimes = [
          'image/png', 'image/jpeg', 'image/webp', 'image/heic', 'image/heif', 'image/gif',
          'audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/aac', 'audio/flac', 'audio/ogg', 'audio/opus', 'audio/webm',
          'video/mp4', 'video/mpeg', 'video/mov', 'video/avi', 'video/flv', 'video/webm', 'video/quicktime',
          'application/pdf', 'text/plain', 'text/html', 'text/css', 'text/javascript', 'application/javascript', 'application/json', 'text/csv', 'text/markdown', 'text/xml', 'application/xml'
        ];

        if (!knownMimes.includes(mimeType)) {
          if (mimeType.startsWith('text/')) {
            mimeType = 'text/plain';
          } else {
            // Unrecognized custom files like .py/.ts (often text/x-python or text/typescript) should fallback to text/plain
            const isTextLike = /\.(txt|md|py|js|ts|tsx|jsx|html|css|json|csv|log|yaml|yml|ini|conf|sql|sh|bash)$/i.test(att.name);
            mimeType = isTextLike ? 'text/plain' : 'application/octet-stream';
          }
        }

        const base64Data = att.data.includes(',') ? att.data.split(',')[1] : att.data;
        currentParts.push({
          inlineData: {
            mimeType: mimeType,
            data: base64Data
          }
        });
      }
    }

    if (message) {
      currentParts.push({ text: message });
    }

    const formattedHistory = history.map((h: any) => ({
      role: h.role,
      parts: [{ text: h.content }]
    }));

    const contents = [
      ...formattedHistory,
      {
        role: 'user',
        parts: currentParts
      }
    ];

    const selectedGeminiModel = geminiModel || 'gemini-3.5-flash';
    const response = await activeAi.models.generateContent({
      model: selectedGeminiModel,
      contents,
      config: {
        systemInstruction: finalSystemInstruction,
        temperature: temperature,
        tools: [{ googleSearch: {} }]
      }
    });

    // Extract search grounding sources
    const searchSources = response.candidates?.[0]?.groundingMetadata?.groundingChunks?.map((chunk: any) => ({
      title: chunk.web?.title || chunk.web?.name || 'Web Source',
      url: chunk.web?.uri || chunk.web?.url || ''
    })).filter((source: any) => source.url) || [];

    res.json({ 
      text: response.text || '',
      timestamp: Date.now(),
      searchSources
    });

  } catch (error: any) {
    console.error('Chat API Error:', error);
    const errMsg = error.message || error.toString() || '';
    const isQuotaError = errMsg.toLowerCase().includes('quota') || 
                         errMsg.toLowerCase().includes('429') || 
                         errMsg.toLowerCase().includes('resource_exhausted') || 
                         errMsg.toLowerCase().includes('limit');
    
    if (isQuotaError) {
      const bilingualError = `⚠️ **API Quota Exceeded (429 Resource Exhausted) / تم تجاوز حد الحصة المتاحة**

The public shared Gemini API quota has been exhausted. You can easily resolve this by:
1. **Waiting a minute**: Some rate limits are evaluated per minute and reset quickly.
2. **Connecting your own free API Key**: Open the **Workspace & API Integrations Hub (Connections tab)** at the top of the sidebar and insert your custom Gemini API key. You can generate a free key instantly at [Google AI Studio](https://aistudio.google.com/).

---

**تم استهلاك حصة الاستخدام المشتركة لـ Gemini API. يمكنك حل هذه المشكلة بسهولة عن طريق:**
1. **الانتظار لمدة دقيقة واحدة**: بعض القيود يتم تقييمها في الدقيقة وتُعاد تهيئتها بسرعة.
2. **ربط مفتاح API الخاص بك مجاناً**: افتح **بوابة الربط والخدمات الذكية (Connections)** في أعلى الشريط الجانبي وضَع مفتاح Gemini الخاص بك. يمكنك الحصول عليه فوراً مجاناً من [Google AI Studio](https://aistudio.google.com/).`;
      return res.status(429).json({ error: bilingualError });
    }

    res.status(500).json({ 
      error: error.message || 'An error occurred during text generation.' 
    });
  }
});

// Analyze current user message to see if there is any user detail/fact to auto-save to memory
app.post('/api/analyze-memory', async (req, res) => {
  try {
    const { message, existingKnowledge = [] } = req.body;
    if (!message || !apiKey) {
      return res.json({ facts: [] });
    }

    // Ask Gemini to extract new personal facts about the user from the current input message
    const extractionPrompt = `You are a user profile analyzer. Look at the following message from the user and extract any new concrete personal facts or preferences about them (e.g. they write Python, they live in Dubai, they prefer short answers, they have a cat, etc.).
    
Message: "${message}"

Current Known Facts:
${existingKnowledge.map((k: any) => `- ${k.fact}`).join('\n')}

Rules:
1. Extract only REAL, meaningful personal details or preferences. Do not extract temporary greetings or fleeting statements.
2. If it is already known or redundant, DO NOT extract it.
3. Return the result in a clean JSON format. An array of objects with the key 'fact' and 'category' (e.g. 'preference', 'tech', 'bio', 'location').
4. If no new facts can be extracted, return an empty array.
5. Strictly return JSON ONLY. No markdown wrappers.`;

    const response = await ai.models.generateContent({
      model: 'gemini-3.5-flash',
      contents: extractionPrompt,
      config: {
        responseMimeType: 'application/json',
        temperature: 0.2
      }
    });

    try {
      const extracted = JSON.parse(response.text || '{}');
      const facts = Array.isArray(extracted) ? extracted : (extracted.facts || []);
      res.json({ facts });
    } catch (e) {
      res.json({ facts: [] });
    }
  } catch (err) {
    console.error('Memory Analysis Error:', err);
    res.json({ facts: [] });
  }
});

// Setup Vite Dev Server / Static Hosting based on Environment
async function start() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server listening on http://0.0.0.0:${PORT}`);
  });
}

start();
