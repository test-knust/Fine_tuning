# Lab: Local LLM Engineering

## Prerequisites
- **Python 3.10+ installed**
- **~10GB Free Disk Space**
- **RAM:** 8GB minimum (16GB recommended)

---

## Part 1: LM Studio (Visual Exploration)
**Objective:** Understand quantization and model parameters.

### Download & Install
- Visit **lmstudio.ai** and install LM Studio.

### Search
1. Click the magnifying glass icon.
2. Search for **Llama 3** or **Mistral v0.3**.

### Select Quantization
- Choose **Q4_K_M (Recommended)** in the right panel.
- Note the file size (~4–5 GB).
- Click **Download**.

### Chat
1. Click the **Chat** icon (speech bubble).
2. Select your downloaded model from the top dropdown.
3. **Task:** Ask the model: *"Explain quantum entanglement to a 5-year-old."*

### Server Mode
1. Click the **<-> (Developer/Server)** icon on the left.
2. Click **Start Server**.
3. Note the URL: **http://localhost:1234**.

> Keep this running for **Part 3**!

---

## Part 2: Ollama (The Developer Way)
**Objective:** Use CLI tools and create custom model personas.

### Download & Install
- Visit **ollama.com** and install Ollama.

### Terminal Basics
1. Open your terminal (Command Prompt/PowerShell on Windows).
2. Run:
   ```bash
   ollama run llama3
   ```
   (This pulls the model automatically.)
3. Chat briefly, then type **/bye** to exit.

### Creating a "Modelfile" (Custom Persona)
1. Create a file named **Modelfile** (no extension).
2. Paste the following content:
   ```
   FROM llama3
   SYSTEM "You are a grumpy senior engineer. You give correct code answers but complain about the user's lack of knowledge. Keep answers concise."
   PARAMETER temperature 0.7
   ```

### Build Your Custom Model
```bash
ollama create grumpy-coder -f Modelfile
```

### Run It
```bash
ollama run grumpy-coder
```

**Task:** Ask it *"How do I center a div?"* and observe the attitude.

---

## Part 3: Python Integration
**Objective:** Control LLMs via code.

### Install Libraries
```bash
pip install ollama openai
```

### Run Scripts
- Open **ollama_script.py** to see how to interact with Ollama programmatically.
- Open **lmstudio_api.py** to see how to swap OpenAI's GPT‑4 for your local LM Studio model.

---

## Part 4: Mini‑RAG (Retrieval Augmented Generation)
**Objective:** Build a system that "reads" a document before answering.

We will not use complex vector databases today. We will build a simple RAG using basic math.

1. Run **tiny_rag.py**.
2. **Challenge:** Modify `tiny_rag.py` to read from a `.txt` file on your computer instead of the hardcoded sentences.

---

