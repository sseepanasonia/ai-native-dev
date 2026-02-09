# LLM and OpenAI SDK Comprehensive Guide

## Table of Contents
1. [Understanding Large Language Models (LLMs)](#understanding-large-language-models-llms)
2. [LLM Core Concepts](#llm-core-concepts)
3. [OpenAI SDK Overview](#openai-sdk-overview)
4. [Chat Completions API](#chat-completions-api)
5. [Completions API (Legacy)](#completions-api-legacy)
6. [Assistants API](#assistants-api)
7. [API Comparison](#api-comparison)
8. [Advanced Features](#advanced-features)
9. [Best Practices](#best-practices)
10. [Cost Optimization](#cost-optimization)
11. [Common Patterns](#common-patterns)

---

## Understanding Large Language Models (LLMs)

### What are LLMs?

**Large Language Models** are AI systems trained on vast amounts of text data to understand and generate human-like text. They use deep learning architectures (primarily Transformers) to predict the next token in a sequence.

### How LLMs Work

```
Input: "The capital of France is"
        ↓
    [Tokenization]
        ↓
    ["The", "capital", "of", "France", "is"]
        ↓
    [Token IDs: 464, 6139, 315, 4501, 318]
        ↓
    [Model Processing]
        ↓
    [Probability Distribution]
        ↓
    "Paris" (highest probability)
```

### Key Characteristics

1. **Pre-trained**: Trained on massive text corpora (books, websites, code)
2. **Few-shot Learning**: Can learn from examples without retraining
3. **Context Window**: Limited memory (tokens it can process at once)
4. **Probabilistic**: Generates based on statistical patterns
5. **General Purpose**: Can handle various tasks without task-specific training

### LLM Generations

| Generation | Models | Key Features | Context Window |
|------------|--------|--------------|----------------|
| **GPT-2** (2019) | GPT-2 | 1.5B parameters | 1K tokens |
| **GPT-3** (2020) | GPT-3, Davinci | 175B parameters | 2K-4K tokens |
| **GPT-3.5** (2022) | ChatGPT, Turbo | Chat-optimized | 4K-16K tokens |
| **GPT-4** (2023) | GPT-4, GPT-4 Turbo | Multimodal, reasoning | 8K-128K tokens |
| **GPT-4o** (2024+) | GPT-4o, GPT-4o mini | Fast, efficient | 128K tokens |

---

## LLM Core Concepts

### 1. Tokens

**Tokens** are the basic units LLMs process. One token ≈ 4 characters or ¾ of a word.

```python
# Example tokenization
"Hello, world!" → ["Hello", ",", " world", "!"]  # 4 tokens
"GPT-4 is amazing" → ["GPT", "-", "4", " is", " amazing"]  # 5 tokens

# English text: ~750 words = ~1000 tokens
# Code: varies widely by language and style
```

**Why Tokens Matter:**
- Pricing is based on tokens
- Context limits are in tokens
- Performance measured in tokens/second

### 2. Context Window

The **maximum number of tokens** an LLM can process in a single interaction (input + output).

```
┌─────────────────────────────────┐
│   Context Window (128K tokens)  │
├─────────────────────────────────┤
│  System Prompt:    500 tokens   │
│  User Input:     2,000 tokens   │
│  Conversation:  10,000 tokens   │
│  Available:    115,500 tokens   │
└─────────────────────────────────┘
```

**Implications:**
- Longer context = more memory but slower/expensive
- Context overflow = truncation or error
- Efficient prompting saves tokens

### 3. Temperature

Controls **randomness** in generation (0.0 to 2.0).

```python
# Temperature comparison
prompt = "The sky is"

# Temperature 0.0 (Deterministic)
response = "blue."  # Most probable, consistent

# Temperature 0.7 (Balanced)
response = "a beautiful shade of blue today."  # Creative but reasonable

# Temperature 1.5 (Creative)
response = "an infinite canvas painted with clouds."  # High creativity

# Temperature 2.0 (Random)
response = "purple dancing elephants"  # Potentially nonsensical
```

**When to Use:**
- `0.0-0.3`: Facts, code, math, translations
- `0.4-0.7`: Balanced tasks, general chat
- `0.8-1.2`: Creative writing, brainstorming
- `1.3-2.0`: Experimental, random outputs

### 4. Top P (Nucleus Sampling)

Alternative to temperature. **Samples from top P probability mass**.

```python
# Example probability distribution
word_probs = {
    "blue": 0.60,      # Cumulative: 0.60
    "clear": 0.20,     # Cumulative: 0.80
    "gray": 0.10,      # Cumulative: 0.90
    "beautiful": 0.05, # Cumulative: 0.95
    "pink": 0.03,      # Cumulative: 0.98
    # ... more options
}

# top_p = 0.8
# Considers: "blue", "clear" (80% of probability mass)

# top_p = 0.95
# Considers: "blue", "clear", "gray", "beautiful"
```

**Best Practice:** Use temperature OR top_p, not both.

### 5. Max Tokens

**Maximum tokens** in the generated response.

```python
# Setting limits
max_tokens = 100  # Response capped at ~75 words

# Use cases:
# Short answers: 50-150 tokens
# Paragraphs: 200-500 tokens
# Long form: 1000+ tokens
# Max allowed: Varies by model (4096-16384)
```

### 6. Stop Sequences

Tokens that **stop generation** when encountered.

```python
# Example: Generate Q&A pairs
stop = ["\n\nQ:", "\n\nHuman:", "---"]

# Generation stops when any stop sequence appears
response = "A: Paris is the capital.\n\nQ:"  # Stops here
```

### 7. Frequency & Presence Penalty

Control **repetition** in responses.

```python
# Frequency Penalty (-2.0 to 2.0)
# Reduces likelihood of repeating tokens based on frequency
frequency_penalty = 0.5  # Mild discouragement
frequency_penalty = 1.5  # Strong discouragement

# Presence Penalty (-2.0 to 2.0)
# Reduces likelihood of any token that appeared before
presence_penalty = 0.5  # Encourages new topics

# Example effect:
# Low penalties: "The cat sat. The cat ate. The cat slept."
# High penalties: "The cat sat. It ate. Then slept peacefully."
```

### 8. System Messages

Instructions that **shape model behavior** (chat models only).

```python
# Examples of effective system messages
system_messages = {
    "code_expert": "You are a senior Python developer. Provide clean, well-documented code following PEP 8.",
    
    "concise_assistant": "Respond in 2-3 sentences maximum. Be direct and clear.",
    
    "json_responder": "Always respond with valid JSON. No additional text.",
    
    "educator": "You are a patient teacher. Explain concepts step-by-step with examples.",
}
```

---

## OpenAI SDK Overview

### Installation

```bash
# Python
pip install openai

# With specific version
pip install openai==1.54.0

# With UV
uv add openai
```

### Basic Setup

```python
from openai import OpenAI
import os

# Initialize client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    # Optional: customize
    timeout=30.0,
    max_retries=3,
)

# Test connection
models = client.models.list()
print(f"Available models: {len(models.data)}")
```

### SDK Structure

```
OpenAI Client
├── chat.completions      # Chat-based generation
├── completions           # Legacy text completion
├── embeddings            # Text embeddings
├── images                # Image generation (DALL-E)
├── audio                 # Speech-to-text, text-to-speech
├── files                 # File management
├── fine_tuning           # Custom model training
├── assistants            # Assistants API
├── threads               # Conversation threads
└── models                # Model information
```

### Authentication

```python
# Method 1: Environment variable (recommended)
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Method 2: Direct (not recommended)
client = OpenAI(api_key="sk-...")

# Method 3: Azure OpenAI
from openai import AzureOpenAI
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Method 4: Custom base URL (local models, proxies)
client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:1234/v1"
)
```

---

## Chat Completions API

### Overview

The **modern, recommended API** for conversational interactions. Supports system messages, multi-turn conversations, and function calling.

### Basic Usage

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

### Message Roles

```python
messages = [
    # System: Sets assistant behavior/personality
    {
        "role": "system",
        "content": "You are a expert Python developer with 10 years experience."
    },
    
    # User: Human input
    {
        "role": "user",
        "content": "How do I handle exceptions in Python?"
    },
    
    # Assistant: AI responses (for conversation history)
    {
        "role": "assistant",
        "content": "You can use try-except blocks..."
    },
    
    # User: Follow-up
    {
        "role": "user",
        "content": "Can you show an example?"
    }
]
```

### Response Structure

```python
response = client.chat.completions.create(...)

# Access response components
message = response.choices[0].message
content = message.content                # Text response
role = message.role                      # "assistant"
finish_reason = response.choices[0].finish_reason  # "stop", "length", etc.

# Metadata
model = response.model                   # Model used
tokens = response.usage.total_tokens     # Token count
prompt_tokens = response.usage.prompt_tokens
completion_tokens = response.usage.completion_tokens

# Example output
"""
ChatCompletion(
    id='chatcmpl-123',
    choices=[
        Choice(
            finish_reason='stop',
            index=0,
            message=ChatCompletionMessage(
                content='Python is a high-level...',
                role='assistant'
            )
        )
    ],
    created=1677652288,
    model='gpt-4o-mini',
    usage=CompletionUsage(
        completion_tokens=150,
        prompt_tokens=20,
        total_tokens=170
    )
)
"""
```

### Streaming Responses

```python
# Stream tokens as they're generated
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

print()  # New line at end
```

### Advanced Streaming with Error Handling

```python
def stream_chat(messages: list) -> str:
    """Stream chat with proper error handling"""
    full_response = ""
    
    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
            timeout=30
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        
        print()  # New line
        return full_response
        
    except Exception as e:
        print(f"\nError: {e}")
        return full_response

# Usage
response = stream_chat([
    {"role": "user", "content": "Explain async programming"}
])
```

### Multi-turn Conversations

```python
class ChatSession:
    """Manage conversation state"""
    
    def __init__(self, system_message: str = "You are helpful"):
        self.messages = [{"role": "system", "content": system_message}]
        self.client = OpenAI()
    
    def send(self, user_message: str) -> str:
        """Send message and get response"""
        # Add user message
        self.messages.append({"role": "user", "content": user_message})
        
        # Get response
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages
        )
        
        # Add assistant response to history
        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
    
    def clear(self):
        """Reset conversation, keeping system message"""
        self.messages = self.messages[:1]

# Usage
chat = ChatSession("You are a Python expert")
print(chat.send("What is a decorator?"))
print(chat.send("Can you show an example?"))  # Remembers context
```

### Function Calling

Enable the model to call functions with structured parameters.

```python
# Define functions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g., San Francisco"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Create completion with tools
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto"
)

# Check if function was called
message = response.choices[0].message
if message.tool_calls:
    tool_call = message.tool_calls[0]
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    
    print(f"Function: {function_name}")
    print(f"Arguments: {arguments}")
    # {"location": "Paris", "unit": "celsius"}
```

### JSON Mode

Force the model to return valid JSON.

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You respond with JSON only"},
        {"role": "user", "content": "Give me info about Python"}
    ],
    response_format={"type": "json_object"}
)

data = json.loads(response.choices[0].message.content)
print(data)
# {"name": "Python", "type": "programming language", ...}
```

### Vision (Multimodal)

Process images with GPT-4 Vision models.

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg",
                        # Or use base64: "data:image/jpeg;base64,..."
                        "detail": "high"  # "low", "high", or "auto"
                    }
                }
            ]
        }
    ],
    max_tokens=500
)

print(response.choices[0].message.content)
```

---

## Completions API (Legacy)

### Overview

The **original text completion API**. Still useful for specific use cases but generally superseded by Chat Completions.

### Basic Usage

```python
response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt="Write a tagline for an ice cream shop:",
    max_tokens=50,
    temperature=0.8
)

print(response.choices[0].text)
```

### Differences from Chat Completions

| Feature | Completions API | Chat Completions API |
|---------|----------------|---------------------|
| **Input** | Plain text string | Structured messages |
| **System messages** | No | Yes |
| **Conversation** | Manual | Built-in |
| **Models** | Limited (instruct models) | All modern models |
| **Recommended** | No | Yes |

### When to Use Completions API

```python
# ✅ Good use cases:
# - Simple text completion
# - One-off generations
# - Legacy code maintenance

# Example: Code completion
prompt = """
def fibonacci(n):
    '''Generate fibonacci sequence'''
"""

response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt,
    max_tokens=150,
    stop=["\n\n", "def "]
)

print(response.choices[0].text)
```

### Migration from Completions to Chat

```python
# BEFORE: Completions API
response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt="Translate 'Hello' to French:",
    max_tokens=10
)
result = response.choices[0].text

# AFTER: Chat Completions API
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Translate 'Hello' to French:"}],
    max_tokens=10
)
result = response.choices[0].message.content
```

---

## Assistants API

### Overview

**High-level API** for building stateful, multi-turn applications with persistent threads, built-in tools, and file handling.

### Key Concepts

```
Assistant (Persistent)
    ↓
Thread (Conversation)
    ↓
Messages (Exchanges)
    ↓
Runs (Execution)
    ↓
Responses
```

### Creating an Assistant

```python
# Create assistant with tools
assistant = client.beta.assistants.create(
    name="Python Tutor",
    instructions="You are a patient Python tutor. Explain concepts clearly with examples.",
    model="gpt-4o-mini",
    tools=[
        {"type": "code_interpreter"},  # Execute Python code
        {"type": "file_search"}         # Search uploaded files
    ]
)

print(f"Assistant ID: {assistant.id}")
```

### Using Assistants

```python
# 1. Create a thread (conversation)
thread = client.beta.threads.create()

# 2. Add a message
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Explain list comprehensions"
)

# 3. Run the assistant
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)

# 4. Wait for completion
import time
while run.status in ["queued", "in_progress"]:
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    time.sleep(0.5)

# 5. Get response
if run.status == "completed":
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    latest_message = messages.data[0]
    print(latest_message.content[0].text.value)
```

### Assistants with Code Interpreter

```python
# Assistant can execute Python code
assistant = client.beta.assistants.create(
    name="Data Analyst",
    instructions="Analyze data and create visualizations",
    model="gpt-4o",
    tools=[{"type": "code_interpreter"}]
)

# Upload data file
file = client.files.create(
    file=open("data.csv", "rb"),
    purpose="assistants"
)

# Run with file
thread = client.beta.threads.create()
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Analyze this sales data and show trends",
    attachments=[{
        "file_id": file.id,
        "tools": [{"type": "code_interpreter"}]
    }]
)

run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id
)

# Get results (may include generated images/charts)
```

### Assistants vs Chat Completions

| Feature | Assistants API | Chat Completions API |
|---------|---------------|---------------------|
| **State Management** | Built-in (threads) | Manual |
| **Code Execution** | Built-in | External |
| **File Handling** | Built-in | Manual |
| **Pricing** | Per-call + storage | Per-token only |
| **Complexity** | Higher abstraction | Lower level |
| **Use Case** | Complex apps | Simple interactions |

---

## API Comparison

### Feature Comparison Matrix

| Feature | Chat Completions | Completions | Assistants |
|---------|-----------------|-------------|------------|
| **Streaming** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Function Calling** | ✅ Yes | ❌ No | ✅ Yes |
| **Conversation State** | Manual | Manual | ✅ Automatic |
| **System Messages** | ✅ Yes | ❌ No | ✅ Yes (instructions) |
| **Vision** | ✅ Yes | ❌ No | ✅ Yes |
| **Code Interpreter** | ❌ No | ❌ No | ✅ Yes |
| **File Search** | ❌ No | ❌ No | ✅ Yes |
| **JSON Mode** | ✅ Yes | ❌ Limited | ✅ Yes |
| **Recommended** | ✅ Yes | ⚠️ Legacy | ✅ Yes (complex) |

### When to Use Each API

```python
# ✅ Chat Completions API
# - Most use cases
# - Conversational AI
# - Function calling
# - Real-time chat
# - Vision tasks
use_chat_completions_for = [
    "Chatbots",
    "Q&A systems",
    "Content generation",
    "Translation",
    "Classification",
    "Function/tool execution"
]

# ⚠️ Completions API (Legacy)
# - Simple text completion
# - Legacy systems
# - Specific instruct models
use_completions_for = [
    "Maintaining old code",
    "Simple one-off completions",
    "When chat format is overkill"
]

# ✅ Assistants API
# - Complex stateful apps
# - Need code execution
# - File processing
# - Multi-step workflows
use_assistants_for = [
    "Data analysis tools",
    "Document processors",
    "Long-running conversations",
    "Research assistants",
    "Tools with persistent state"
]
```

### Cost Comparison

```python
# Pricing varies by model (example rates as of 2024)

# Chat Completions (gpt-4o-mini)
# Input: $0.150 / 1M tokens
# Output: $0.600 / 1M tokens

# Assistants API
# Same per-token pricing PLUS:
# - Code interpreter: $0.03 per session
# - File search: $0.10 per GB per day (storage)

# Example calculation:
tokens_in = 1000
tokens_out = 500

chat_cost = (tokens_in * 0.150 + tokens_out * 0.600) / 1_000_000
# = $0.00045

assistants_cost = chat_cost + 0.03  # If using code interpreter
# = $0.03045
```

---

## Advanced Features

### 1. Rate Limiting & Retries

```python
from openai import OpenAI, RateLimitError, APIError
import time

client = OpenAI(max_retries=3, timeout=30)

def call_with_retry(messages, max_retries=5):
    """Call API with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Rate limited. Waiting {wait_time}s...")
            time.sleep(wait_time)
        except APIError as e:
            print(f"API error: {e}")
            raise

# Usage
response = call_with_retry([
    {"role": "user", "content": "Hello"}
])
```

### 2. Async Operations

```python
from openai import AsyncOpenAI
import asyncio

async_client = AsyncOpenAI()

async def get_completion(prompt: str) -> str:
    """Async completion"""
    response = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

async def batch_completions(prompts: list[str]) -> list[str]:
    """Process multiple prompts concurrently"""
    tasks = [get_completion(p) for p in prompts]
    return await asyncio.gather(*tasks)

# Usage
async def main():
    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?"
    ]
    results = await batch_completions(prompts)
    for prompt, result in zip(prompts, results):
        print(f"Q: {prompt}\nA: {result}\n")

asyncio.run(main())
```

### 3. Token Counting

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in text"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Example
text = "Hello, how are you doing today?"
tokens = count_tokens(text)
print(f"Text: {text}")
print(f"Tokens: {tokens}")

# Estimate cost
def estimate_cost(prompt_tokens: int, completion_tokens: int,
                  model: str = "gpt-4o-mini") -> float:
    """Estimate cost in USD"""
    # Prices as of 2024 (update as needed)
    prices = {
        "gpt-4o-mini": {"input": 0.150, "output": 0.600},
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00}
    }
    
    price = prices.get(model, prices["gpt-4o-mini"])
    cost = (prompt_tokens * price["input"] + 
            completion_tokens * price["output"]) / 1_000_000
    return cost

cost = estimate_cost(1000, 500, "gpt-4o-mini")
print(f"Estimated cost: ${cost:.6f}")
```

### 4. Embeddings

```python
# Generate embeddings for semantic search
def get_embedding(text: str, model="text-embedding-3-small"):
    """Get embedding vector"""
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

# Example: Semantic similarity
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Compare texts
texts = [
    "The cat sits on the mat",
    "A feline rests on the rug",
    "Python is a programming language"
]

embeddings = [get_embedding(t) for t in texts]

# Similarity between first two (similar meaning)
sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
print(f"Similarity 1-2: {sim_1_2:.3f}")  # High similarity

# Similarity between first and third (different meaning)
sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])
print(f"Similarity 1-3: {sim_1_3:.3f}")  # Low similarity
```

### 5. Image Generation (DALL-E)

```python
# Generate images
response = client.images.generate(
    model="dall-e-3",
    prompt="A serene Japanese garden with cherry blossoms",
    size="1024x1024",
    quality="standard",  # or "hd"
    n=1
)

image_url = response.data[0].url
print(f"Generated image: {image_url}")

# Edit images
response = client.images.edit(
    image=open("original.png", "rb"),
    mask=open("mask.png", "rb"),
    prompt="Add a red hat",
    n=1,
    size="1024x1024"
)

# Create variations
response = client.images.create_variation(
    image=open("original.png", "rb"),
    n=2,
    size="1024x1024"
)
```

### 6. Speech (Text-to-Speech & Speech-to-Text)

```python
# Text to Speech
response = client.audio.speech.create(
    model="tts-1",  # or "tts-1-hd" for higher quality
    voice="alloy",  # alloy, echo, fable, onyx, nova, shimmer
    input="Hello! This is a test of text to speech."
)

# Save audio
with open("speech.mp3", "wb") as f:
    f.write(response.content)

# Speech to Text (Whisper)
audio_file = open("speech.mp3", "rb")
transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="text"  # or "json", "srt", "vtt"
)

print(transcription)

# Translation (to English)
translation = client.audio.translations.create(
    model="whisper-1",
    file=open("spanish_audio.mp3", "rb")
)
print(translation.text)
```

---

## Best Practices

### 1. Prompt Engineering

```python
# ❌ Bad: Vague prompt
prompt = "Tell me about dogs"

# ✅ Good: Specific, structured prompt
prompt = """
Task: Write a 3-paragraph essay about golden retrievers
Format: 
- Paragraph 1: History and origin
- Paragraph 2: Characteristics and temperament
- Paragraph 3: Care requirements
Tone: Informative but friendly
Length: Approximately 200 words
"""

# ✅ Better: Few-shot learning
messages = [
    {"role": "system", "content": "You format responses as structured data"},
    {"role": "user", "content": "Product: iPhone\nPrice: $999"},
    {"role": "assistant", "content": '{"product": "iPhone", "price": 999}'},
    {"role": "user", "content": "Product: MacBook\nPrice: $1299"},
    {"role": "assistant", "content": '{"product": "MacBook", "price": 1299}'},
    {"role": "user", "content": "Product: AirPods\nPrice: $249"}
]
```

### 2. Error Handling

```python
from openai import (
    OpenAI,
    APIError,
    APIConnectionError,
    RateLimitError,
    AuthenticationError
)

def robust_completion(messages: list, max_retries: int = 3):
    """Production-ready completion with error handling"""
    client = OpenAI()
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                timeout=30
            )
            return response
            
        except AuthenticationError:
            # Don't retry auth errors
            raise ValueError("Invalid API key")
            
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise Exception("Rate limit exceeded")
            wait_time = 2 ** attempt
            time.sleep(wait_time)
            
        except APIConnectionError as e:
            if attempt == max_retries - 1:
                raise Exception("Connection failed")
            time.sleep(1)
            
        except APIError as e:
            # Server errors
            if e.status_code >= 500:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
            raise Exception(f"API error: {e}")
            
        except Exception as e:
            raise Exception(f"Unexpected error: {e}")
    
    raise Exception("Max retries exceeded")
```

### 3. Configuration Management

```python
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class OpenAIConfig:
    """Centralized configuration"""
    api_key: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30
    max_retries: int = 3
    
    @classmethod
    def from_env(cls):
        """Load from environment"""
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
        )
    
    def get_client(self):
        """Create configured client"""
        return OpenAI(
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries
        )

# Usage
config = OpenAIConfig.from_env()
client = config.get_client()
```

### 4. Input Validation

```python
from pydantic import BaseModel, Field, validator

class ChatInput(BaseModel):
    """Validated chat input"""
    message: str = Field(..., min_length=1, max_length=4000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=500, ge=1, le=4096)
    
    @validator('message')
    def sanitize_message(cls, v):
        # Remove potentially harmful content
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty")
        # Add more sanitization as needed
        return v

# Usage
try:
    input_data = ChatInput(
        message=user_input,
        temperature=0.8,
        max_tokens=1000
    )
    # Proceed with validated data
except ValueError as e:
    print(f"Invalid input: {e}")
```

### 5. Logging & Monitoring

```python
import logging
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def logged_completion(messages: list, **kwargs):
    """Completion with comprehensive logging"""
    start_time = datetime.now()
    
    try:
        # Log request
        logger.info(f"API Request: model={kwargs.get('model')}")
        logger.debug(f"Messages: {json.dumps(messages)}")
        
        # Make request
        response = client.chat.completions.create(
            messages=messages,
            **kwargs
        )
        
        # Log response
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"API Response: tokens={response.usage.total_tokens}, "
                   f"duration={duration:.2f}s")
        logger.debug(f"Response: {response.choices[0].message.content[:100]}")
        
        return response
        
    except Exception as e:
        logger.error(f"API Error: {type(e).__name__}: {e}")
        raise

# Usage
response = logged_completion(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4o-mini"
)
```

### 6. Caching

```python
from functools import lru_cache
import hashlib
import json

class ResponseCache:
    """Simple response cache"""
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
    
    def _get_key(self, messages: list, **kwargs) -> str:
        """Generate cache key"""
        data = json.dumps({"messages": messages, **kwargs}, sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()
    
    def get(self, messages: list, **kwargs):
        """Get cached response"""
        key = self._get_key(messages, **kwargs)
        return self.cache.get(key)
    
    def set(self, messages: list, response, **kwargs):
        """Cache response"""
        if len(self.cache) >= self.max_size:
            # Simple eviction: remove oldest
            self.cache.pop(next(iter(self.cache)))
        
        key = self._get_key(messages, **kwargs)
        self.cache[key] = response

# Usage
cache = ResponseCache()

def cached_completion(messages: list, **kwargs):
    """Check cache before API call"""
    # Try cache first
    cached = cache.get(messages, **kwargs)
    if cached:
        logger.info("Cache hit")
        return cached
    
    # Make API call
    response = client.chat.completions.create(messages=messages, **kwargs)
    
    # Cache response
    cache.set(messages, response, **kwargs)
    
    return response
```

### 7. Context Window Management

```python
def truncate_conversation(messages: list, max_tokens: int = 4000) -> list:
    """Keep conversation within token limit"""
    # Always keep system message
    system_messages = [m for m in messages if m["role"] == "system"]
    other_messages = [m for m in messages if m["role"] != "system"]
    
    # Count tokens (simplified - use tiktoken for accuracy)
    total_tokens = sum(len(m["content"].split()) for m in messages)
    
    # Truncate older messages if needed
    while total_tokens > max_tokens and len(other_messages) > 1:
        # Remove oldest non-system message
        removed = other_messages.pop(0)
        total_tokens -= len(removed["content"].split())
    
    return system_messages + other_messages

# Usage in chat
class ManagedChat:
    def __init__(self, max_context_tokens: int = 4000):
        self.messages = [{"role": "system", "content": "You are helpful"}]
        self.max_tokens = max_context_tokens
    
    def send(self, message: str) -> str:
        self.messages.append({"role": "user", "content": message})
        
        # Truncate if needed
        self.messages = truncate_conversation(self.messages, self.max_tokens)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages
        )
        
        assistant_msg = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_msg})
        
        return assistant_msg
```

---

## Cost Optimization

### 1. Model Selection

```python
# Choose appropriate model for task complexity

tasks = {
    "simple_classification": "gpt-4o-mini",  # Cheapest
    "general_chat": "gpt-4o-mini",
    "complex_reasoning": "gpt-4o",
    "code_generation": "gpt-4o",
    "creative_writing": "gpt-4-turbo",
}

def get_optimal_model(task_type: str) -> str:
    """Select cost-effective model"""
    return tasks.get(task_type, "gpt-4o-mini")

# Example: Try cheaper model first
def smart_completion(prompt: str, task_complexity: str = "low"):
    """Use cheaper model when possible"""
    model = "gpt-4o-mini" if task_complexity == "low" else "gpt-4o"
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response
```

### 2. Token Optimization

```python
# ❌ Wasteful
messages = [
    {"role": "system", "content": "You are a very helpful, kind, and friendly AI assistant that always tries to help users with their questions and provides detailed, comprehensive answers with lots of examples and context."},
    {"role": "user", "content": "What is 2+2?"}
]

# ✅ Optimized
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
]

# Optimize prompts
def optimize_prompt(prompt: str) -> str:
    """Remove unnecessary words"""
    # Remove filler words
    fillers = ["actually", "basically", "literally", "very", "really"]
    for filler in fillers:
        prompt = prompt.replace(f" {filler} ", " ")
    
    # Remove extra whitespace
    import re
    prompt = re.sub(r'\s+', ' ', prompt).strip()
    
    return prompt
```

### 3. Batch Processing

```python
async def batch_with_delay(prompts: list[str], delay: float = 0.1):
    """Process batch with rate limiting"""
    results = []
    
    for prompt in prompts:
        response = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        results.append(response.choices[0].message.content)
        
        # Rate limiting
        await asyncio.sleep(delay)
    
    return results
```

### 4. Response Reuse

```python
class SmartCache:
    """Cache with similarity matching"""
    def __init__(self):
        self.cache = {}
    
    def find_similar(self, prompt: str, threshold: float = 0.9) -> Optional[str]:
        """Find similar cached prompts"""
        # Simple word-based similarity (use embeddings for better results)
        prompt_words = set(prompt.lower().split())
        
        for cached_prompt, response in self.cache.items():
            cached_words = set(cached_prompt.lower().split())
            similarity = len(prompt_words & cached_words) / len(prompt_words | cached_words)
            
            if similarity >= threshold:
                return response
        
        return None
    
    def get_or_create(self, prompt: str) -> str:
        """Get from cache or create new"""
        # Exact match
        if prompt in self.cache:
            return self.cache[prompt]
        
        # Similar match
        similar = self.find_similar(prompt)
        if similar:
            return similar
        
        # Create new
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content
        
        self.cache[prompt] = result
        return result
```

---

## Common Patterns

### 1. Chain of Thought

```python
def chain_of_thought(question: str) -> str:
    """Encourage step-by-step reasoning"""
    messages = [
        {"role": "system", "content": "Think step by step and show your reasoning."},
        {"role": "user", "content": f"{question}\n\nLet's approach this step by step:"}
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3  # Lower for logical tasks
    )
    
    return response.choices[0].message.content

# Example
answer = chain_of_thought("If a train travels 60 mph for 2.5 hours, how far does it go?")
```

### 2. Few-Shot Learning

```python
def few_shot_classifier(text: str, examples: list) -> str:
    """Classify using examples"""
    messages = [{"role": "system", "content": "Classify the sentiment"}]
    
    # Add examples
    for example in examples:
        messages.append({"role": "user", "content": example["text"]})
        messages.append({"role": "assistant", "content": example["label"]})
    
    # Add actual query
    messages.append({"role": "user", "content": text})
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3
    )
    
    return response.choices[0].message.content

# Usage
examples = [
    {"text": "I love this product!", "label": "positive"},
    {"text": "This is terrible.", "label": "negative"},
    {"text": "It's okay, nothing special.", "label": "neutral"}
]

result = few_shot_classifier("Amazing experience!", examples)
```

### 3. Self-Consistency

```python
def self_consistent_answer(question: str, n: int = 5) -> str:
    """Generate multiple answers and pick most common"""
    from collections import Counter
    
    answers = []
    for _ in range(n):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": question}],
            temperature=0.7  # Some randomness
        )
        answers.append(response.choices[0].message.content)
    
    # Find most common answer
    counter = Counter(answers)
    most_common = counter.most_common(1)[0][0]
    
    return most_common
```

### 4. Iterative Refinement

```python
def iterative_refinement(initial_prompt: str, iterations: int = 3) -> str:
    """Refine response over multiple iterations"""
    current_response = ""
    
    for i in range(iterations):
        if i == 0:
            prompt = initial_prompt
        else:
            prompt = f"Improve this response:\n\n{current_response}\n\nMake it more concise and clear."
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        current_response = response.choices[0].message.content
    
    return current_response
```

### 5. Structured Extraction

```python
def extract_structured_data(text: str) -> dict:
    """Extract structured information"""
    prompt = f"""
Extract key information from this text as JSON:

Text: {text}

Return JSON with fields: name, email, phone, company
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

# Example
text = "Contact John Doe at john@example.com or 555-1234. He works at Acme Corp."
data = extract_structured_data(text)
print(data)
# {"name": "John Doe", "email": "john@example.com", "phone": "555-1234", "company": "Acme Corp"}
```

---

## Summary

### Quick Decision Guide

```python
# Choose your API:

# 🎯 Use Chat Completions API when:
✅ Building conversational applications
✅ Need modern features (function calling, vision)
✅ Want streaming support
✅ Working with any recent model
# This is the DEFAULT choice for 95% of use cases

# 📜 Use Completions API when:
⚠️ Maintaining legacy code
⚠️ Using specific instruct models
⚠️ Simple one-off completions
# Generally avoid for new projects

# 🤖 Use Assistants API when:
✅ Need persistent conversation state
✅ Require code execution
✅ Processing files/documents
✅ Building complex multi-step workflows
✅ Want built-in tools and retrieval
# Great for complex applications, but more overhead
```

### Best Practices Checklist

- ✅ Use environment variables for API keys
- ✅ Implement retry logic with exponential backoff
- ✅ Validate and sanitize user inputs
- ✅ Monitor token usage and costs
- ✅ Cache responses when appropriate
- ✅ Use appropriate temperature for task
- ✅ Choose cost-effective models
- ✅ Implement comprehensive error handling
- ✅ Log requests for debugging
- ✅ Manage context window size
- ✅ Use streaming for long responses
- ✅ Test with edge cases

---

## Additional Resources

- **OpenAI Platform**: https://platform.openai.com/
- **API Reference**: https://platform.openai.com/docs/api-reference
- **Cookbook**: https://github.com/openai/openai-cookbook
- **Community Forum**: https://community.openai.com/
- **Status Page**: https://status.openai.com/

---

*Last Updated: February 2026*
