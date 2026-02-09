# Basic LLM Invocation with LangChain + Gemini

A FastAPI application demonstrating basic LLM invocation workflow with Google's Gemini AI model.

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) package manager
- Google Gemini API key

## Installation

1. Navigate to the project directory:

   **For Linux:**

   ```bash
   cd demo-01-setting-up-and-calling-an-llm-in-langchain
   ```

   **For Windows:**

   ```cmd
   cd demo-01-setting-up-and-calling-an-llm-in-langchain
   ```

2. Install dependencies using UV:

   **For Linux/Windows (Same command):**

   ```bash
   uv sync
   ```

   This will automatically:
   - Create a virtual environment
   - Install all dependencies from `pyproject.toml`
   - Set up the project environment

## Configuration

1. Create a `.env` file in the project root:

   **For Linux:**

   ```bash
   touch .env
   ```

   **For Windows (PowerShell):**

   ```powershell
   New-Item -Path .env -ItemType File
   ```

   **For Windows (CMD):**

   ```cmd
   type nul > .env
   ```

2. Add your Google Gemini API key to the `.env` file:

   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
   GEMINI_MODEL_NAME=gemini-2.5-flash
   ```

   **Note**:
   To get a Gemini API key:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Sign in with your Google account
   - Create a new API key
   - The GEMINI_MODEL_NAME value can be updated to any supported model. Model names may change over time, so always refer to the latest options in Google’s documentation.

## Running the Application

**For Linux/Windows (Same commands):**

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The application will start on `http://localhost:8000`

**Note**: On Windows, you can use either PowerShell or CMD for these commands.

## API Usage

**Endpoint:** `POST /chat`

**Request:**

```json
{
  "message": "Write a short motivational quote about learning AI."
}
```

**Response:**

```json
{
  "response": "Your AI-generated response here...",
  "model": "gemini-2.0-flash"
}
```

**Interactive Documentation:** http://localhost:8000/docs
