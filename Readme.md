# BIS Standards Recommendation Engine

A Retrieval-Augmented Generation (RAG) system that recommends relevant Bureau of Indian Standards (BIS) based on user queries.

## Quick Start (5 Minutes)

If you just want to get up and running quickly:

```bash
# 1. Navigate to project
cd BIS

# 2. Create and activate virtual environment
python -m venv env
source env/Scripts/activate  # Windows: env\Scripts\activate.bat

# 3. Install everything and cache models
bash setup.sh  # Windows: python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# 4. Add your API key
echo "GROQ_API_KEY=your_key_here" > .env

# 5. Run inference
python inference.py --input data/public_test_set.json --output data/result.json
```

**For detailed setup instructions, see [Installation & Environment Setup](#installation--environment-setup) section below.**

---

## Overview

This project implements an intelligent recommendation engine that:
- Extracts and chunks BIS standards from PDF documents
- Builds semantic embeddings using sentence transformers
- Retrieves relevant standards using FAISS vector search
- Generates recommendations using Groq's LLM API

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BIS RECOMMENDATION ENGINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  INPUT QUERY                                                      │
│      │                                                             │
│      ▼                                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. EMBEDDING STAGE                                       │   │
│  │    - Convert query to vector using BAAI/bge model       │   │
│  │    - Dimension: 384                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│      │                                                             │
│      ▼                                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 2. RETRIEVAL STAGE                                       │   │
│  │    - Search FAISS index for top-5 similar chunks        │   │
│  │    - Extract context from chunks.json                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│      │                                                             │
│      ▼                                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 3. GENERATION STAGE                                      │   │
│  │    - Send query + context to Groq LLM                   │   │
│  │    - Extract standard codes from response               │   │
│  │    - Deduplicate and rank results                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│      │                                                             │
│      ▼                                                             │
│  OUTPUT: Top-5 BIS Standards                                      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
BIS/
├── src/
│   └── build_index.py          # Index building pipeline
├── data/
│   ├── public_test_set.json    # Test queries
│   └── result.json             # Evaluation results
├── inference.py                # Query processing & recommendation
├── eval_script.py              # Evaluation metrics
├── requirements.txt            # Python dependencies
├── index.faiss                 # Pre-built FAISS index
├── chunks.json                 # Extracted standard chunks
└── dataset.pdf                 # BIS standards source document
```

## Installation & Environment Setup

This section provides detailed step-by-step instructions for setting up the Python environment and installing all dependencies.

### Prerequisites

Before starting, ensure you have:
- **Python 3.8+** installed on your system
- **pip** (Python package manager) - usually comes with Python
- **Git** (optional, for cloning the repository)
- **Internet connection** (for downloading dependencies and models)

### Step 1: Navigate to Project Directory

```bash
cd BIS
```

This command moves you into the project folder where all the code and configuration files are located.

### Step 2: Create a Python Virtual Environment

A virtual environment is an isolated Python workspace that prevents dependency conflicts with other projects.

**On Windows (Command Prompt or PowerShell):**
```bash
python -m venv env
```

**On macOS/Linux (Terminal):**
```bash
python3 -m venv env
```

**What this does:**
- Creates a new folder called `env/` in your project directory
- Sets up an isolated Python environment with its own package manager
- Prevents conflicts between project dependencies and system-wide Python packages

### Step 3: Activate the Virtual Environment

Activating the virtual environment ensures all packages are installed in the isolated environment.

**On Windows (Command Prompt):**
```bash
env\Scripts\activate.bat
```

**On Windows (PowerShell):**
```bash
.\env\Scripts\Activate.ps1
```

**On macOS/Linux (Terminal):**
```bash
source env/bin/activate
```

**How to verify activation:**
After running the activation command, your terminal prompt should change to show `(env)` at the beginning:
```
(env) C:\Users\YourName\Desktop\BIS>
```

If you see `(env)` in your prompt, the virtual environment is **active** ✓

### Step 4: Install Dependencies

Now install all required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**What this does:**
- Reads the `requirements.txt` file
- Downloads and installs all specified packages:
  - `pymupdf4llm` - PDF processing
  - `sentence-transformers` - Embedding model
  - `faiss_cpu` - Vector search
  - `groq` - LLM API client
  - `numpy` - Numerical operations
  - `scikit-learn` - Machine learning utilities
  - `python-dotenv` - Environment variable management
  - `chromadb` - Vector database (optional)

**Installation time:** 5-10 minutes (depending on internet speed)

**Verify installation:**
After installation completes, verify all packages are installed:
```bash
pip list
```

You should see all packages from `requirements.txt` in the output.

### Step 5: Pre-Cache the Embedding Model (IMPORTANT)

The embedding model needs to be downloaded before running inference. This step ensures the model is cached locally and won't count against your 5-second latency budget during evaluation.

**Option A: Using the setup.sh script (Recommended for macOS/Linux):**

```bash
bash setup.sh
```

This script automatically:
1. Installs all dependencies (same as Step 4)
2. Downloads and caches the BAAI/bge-small-en-v1.5 embedding model
3. Verifies everything is working correctly

**Option B: Manual model caching (All platforms):**

If `setup.sh` doesn't work on your system, run this Python command:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
```

**What this does:**
- Downloads the embedding model (~100MB)
- Stores it in your local cache directory
- Verifies the model loads correctly

**Expected output:**
```
Downloading model weights...
Model successfully cached locally.
```

**Download time:** 2-5 minutes (depending on internet speed)

**Why this matters:**
- First-time model loading takes ~2-3 seconds
- Pre-caching ensures this happens during setup, not during evaluation
- Your inference latency will be measured from query to result, not including model loading

### Step 6: Configure API Key

Create a `.env` file in the root project directory to store your Groq API key. This is a **critical step** - without it, the inference script will not work.

#### 6.1: Get Your Groq API Key

**Step-by-step guide to obtain your API key:**

1. **Open your web browser** and go to: https://console.groq.com/

2. **Sign up or log in:**
   - If you don't have an account, click "Sign Up"
   - Enter your email address
   - Create a password
   - Verify your email
   - If you already have an account, click "Sign In"

3. **Navigate to API Keys section:**
   - After logging in, look for the menu on the left sidebar
   - Click on "API Keys" or "Keys" option
   - You should see a page titled "API Keys"

4. **Create a new API key:**
   - Click the button that says "Create API Key" or "New API Key"
   - A dialog box will appear
   - Give your key a name (e.g., "BIS Hackathon")
   - Click "Create"

5. **Copy your API key:**
   - Your new API key will be displayed (usually starts with `gsk_`)
   - **IMPORTANT:** Copy the entire key immediately
   - This is the only time you'll see the full key - you cannot view it again
   - If you lose it, you'll need to create a new one

**Example API key format:**
```
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
```

#### 6.2: Create the .env File

Now create a `.env` file in your project root directory with your API key.

**Option A: Using Command Line (Recommended)**

**On Windows (Command Prompt):**
```bash
echo GROQ_API_KEY=your_api_key_here > .env
```

Replace `your_api_key_here` with your actual API key:
```bash
echo GROQ_API_KEY=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 > .env
```

**On Windows (PowerShell):**
```bash
"GROQ_API_KEY=your_api_key_here" | Out-File -Encoding UTF8 .env
```

Replace `your_api_key_here` with your actual API key:
```bash
"GROQ_API_KEY=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" | Out-File -Encoding UTF8 .env
```

**On macOS/Linux:**
```bash
echo "GROQ_API_KEY=your_api_key_here" > .env
```

Replace `your_api_key_here` with your actual API key:
```bash
echo "GROQ_API_KEY=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" > .env
```

**Option B: Manual File Creation**

1. **Open a text editor:**
   - Windows: Notepad, VS Code, or any text editor
   - macOS: TextEdit, VS Code, or any text editor
   - Linux: nano, vim, VS Code, or any text editor

2. **Type the following (replace with your actual key):**
   ```
   GROQ_API_KEY=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
   ```

3. **Save the file:**
   - Save as `.env` (note the dot at the beginning)
   - Save location: **Project root directory** (same folder as `inference.py`)
   - File encoding: UTF-8

**Example `.env` file content:**
```
GROQ_API_KEY=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
```

#### 6.3: Verify Your API Key is Configured

After creating the `.env` file, verify it was created correctly:

**On Windows (Command Prompt):**
```bash
type .env
```

**On Windows (PowerShell):**
```bash
Get-Content .env
```

**On macOS/Linux:**
```bash
cat .env
```

**Expected output:**
```
GROQ_API_KEY=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
```

If you see your API key displayed, the `.env` file is **correctly configured** ✓

#### 6.4: How the API Key is Used

The API key is used by the `inference.py` script to authenticate with Groq's LLM service:

1. **When you run inference:**
   ```bash
   python inference.py --input data/public_test_set.json --output data/result.json
   ```

2. **The script does this automatically:**
   - Reads the `.env` file
   - Extracts your `GROQ_API_KEY`
   - Connects to Groq's API using your key
   - Sends queries to the LLM for processing
   - Returns results

3. **Code that uses your API key** (in `inference.py`):
   ```python
   from dotenv import load_dotenv
   import os
   
   load_dotenv()  # Reads .env file
   GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
   
   from groq import Groq
   client = Groq(api_key=GROQ_API_KEY)  # Uses your key
   ```

#### 6.5: Security & Best Practices

**IMPORTANT Security Notes:**

- DO: Keep your API key private and secure
- DO: Never share your API key with anyone
- DO: Never commit `.env` to version control (it's in `.gitignore`)
- DO: Regenerate your key if you accidentally expose it
- DON'T: Paste your API key in chat, emails, or public forums
- DON'T: Include `.env` in your submission
- DON'T: Hardcode the API key in your Python files

**If you accidentally expose your API key:**
1. Go to https://console.groq.com/keys
2. Delete the exposed key
3. Create a new API key
4. Update your `.env` file with the new key

#### 6.6: Troubleshooting API Key Issues

**Issue: "GROQ_API_KEY not found" error**
- Solution: Ensure `.env` file is in the project root (same folder as `inference.py`)
- Solution: Verify the file is named exactly `.env` (with the dot)
- Solution: Check that the file contains `GROQ_API_KEY=your_key_here`

**Issue: "Invalid API key" error**
- Solution: Verify you copied the entire API key correctly
- Solution: Check for extra spaces or characters in the `.env` file
- Solution: Regenerate a new API key from the Groq console

**Issue: "Authentication failed" error**
- Solution: Verify your API key is still valid (not expired or revoked)
- Solution: Check your internet connection
- Solution: Try creating a new API key

**Issue: "Rate limit exceeded" error**
- Solution: Wait a few minutes before running inference again
- Solution: Check your Groq account for usage limits
- Solution: Contact Groq support if you need higher limits

### Step 7: Verify Complete Setup

Run this command to verify everything is installed correctly:

```bash
python -c "import faiss; import numpy; from sentence_transformers import SentenceTransformer; from groq import Groq; print('✓ All dependencies loaded successfully!')"
```

**Expected output:**
```
✓ All dependencies loaded successfully!
```

If you see this message, your environment is **fully configured** ✓

### Complete Setup Checklist

- [ ] Python 3.8+ installed
- [ ] Navigated to BIS project directory
- [ ] Created virtual environment (`env/` folder exists)
- [ ] Activated virtual environment (prompt shows `(env)`)
- [ ] Installed dependencies (`pip install -r requirements.txt`)
- [ ] Pre-cached embedding model (ran `setup.sh` or manual command)
- [ ] Created `.env` file with GROQ_API_KEY
- [ ] Verified setup (all imports successful)

### Troubleshooting Setup Issues

**Issue: "python: command not found"**
- **Solution:** Python is not in your PATH. Try `python3` instead of `python`

**Issue: "No module named 'venv'"**
- **Solution:** Install venv: `python -m pip install --upgrade pip` then retry

**Issue: Virtual environment won't activate**
- **Solution:** Check file permissions. On Windows, try PowerShell as Administrator

**Issue: "pip: command not found"**
- **Solution:** Use `python -m pip install -r requirements.txt` instead

**Issue: Model download fails**
- **Solution:** Check internet connection. Try again or manually download from Hugging Face

**Issue: GROQ_API_KEY not recognized**
- **Solution:** Ensure `.env` file is in the project root (same folder as `inference.py`)

### Deactivating the Virtual Environment

When you're done working, deactivate the virtual environment:

```bash
deactivate
```

Your prompt will return to normal (no `(env)` prefix).

## Usage

### Building the Index

To rebuild the FAISS index from the dataset PDF:

```bash
python src/build_index.py
```

This will:
- Extract markdown from `dataset.pdf`
- Parse BIS standard codes (IS XXXX format)
- Create semantic embeddings
- Generate `index.faiss` and `chunks.json`

### Running Inference

Process queries and get standard recommendations:

```bash
python inference.py --input data/public_test_set.json --output data/result.json
```

**Input Format** (`public_test_set.json`):
```json
[
  {
    "id": "query_001",
    "query": "Standard for lightweight concrete masonry blocks?",
    "expected_standards": ["IS 2185 (Part 1): 1979", "IS 2185 (Part 2): 1983"]
  }
]
```

**Output Format** (`result.json`):
```json
[
  {
    "id": "query_001",
    "query": "Standard for lightweight concrete masonry blocks?",
    "expected_standards": ["IS 2185 (Part 1): 1979", "IS 2185 (Part 2): 1983"],
    "retrieved_standards": ["IS 2185 (Part 1): 1979", "IS 2185 (Part 2): 1983"],
    "latency_seconds": 2.456
  }
]
```

### Evaluating Results

Compute performance metrics:

```bash
python eval_script.py --results data/result.json
```

**Metrics:**
- **Hit Rate @3**: Percentage of queries with ≥1 correct standard in top 3 results (Target: >80%)
- **MRR @5**: Mean Reciprocal Rank of first correct standard in top 5 (Target: >0.7)
- **Avg Latency**: Average response time per query (Target: <5 seconds)

## Technical Details

### Architecture

1. **Indexing Pipeline**
   - PDF → Markdown extraction (PyMuPDF4LLM)
   - Regex-based standard code parsing
   - Semantic embedding (BAAI/bge-small-en-v1.5)
   - FAISS vector indexing

2. **Retrieval Pipeline**
   - Query embedding
   - Top-5 semantic similarity search
   - Context extraction from chunks

3. **Generation Pipeline**
   - System prompt engineering for strict extraction
   - Groq LLM (llama-3.1-8b-instant)
   - Regex post-processing for standard code extraction
   - Deduplication of results

### Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Embedding Model | BAAI/bge-small-en-v1.5 | Semantic representation |
| Vector DB | FAISS | Fast similarity search |
| LLM | Groq (llama-3.1-8b) | Standard recommendation |
| PDF Processing | PyMuPDF4LLM | Document extraction |

## Dependencies

- `pymupdf4llm` - PDF to Markdown conversion
- `sentence-transformers` - Semantic embeddings
- `faiss_cpu` - Vector similarity search
- `groq` - LLM API client
- `numpy` - Numerical operations
- `python-dotenv` - Environment configuration

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Hit Rate @3 | >80% | ✓ |
| MRR @5 | >0.7 | ✓ |
| Latency | <5 sec | ✓ |

## Submission Guidelines

### Deliverables

1. **Source Code**
   - `src/build_index.py` - Indexing pipeline
   - `inference.py` - Query processing
   - `eval_script.py` - Evaluation metrics

2. **Pre-built Assets**
   - `index.faiss` - FAISS vector index
   - `chunks.json` - Extracted standard chunks

3. **Configuration**
   - `requirements.txt` - Dependencies
   - `.env` - API credentials (not committed)

### Package Structure

```
submission/
├── src/
│   └── build_index.py
├── inference.py
├── eval_script.py
├── requirements.txt
├── index.faiss
├── chunks.json
└── README.md
```

## Evaluation Criteria

### Automated Scoring (40 Points)
- Hit Rate @3: 20 points
- MRR @5: 20 points

### Manual Scoring (60 Points)
- Code quality & documentation: 20 points
- Efficiency & optimization: 20 points
- Innovation & approach: 20 points

## Troubleshooting

**Issue**: `FATAL: Could not load index.faiss`
- **Solution**: Run `python src/build_index.py` to rebuild the index

**Issue**: `API Error during generation`
- **Solution**: Verify GROQ_API_KEY in `.env` file

**Issue**: Low Hit Rate
- **Solution**: Adjust embedding model or retrieval parameters in `inference.py`

## References

- [BIS Standards Database](https://www.bis.gov.in/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [Groq API](https://console.groq.com/)

## License

This project is part of the BIS Standards Recommendation Engine Hackathon.

## Contact

For questions or issues, refer to the hackathon guidelines or contact the organizers.
