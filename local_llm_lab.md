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

### Settings
- Settings (Creativity vs. Logic)
   1. Temperature (0.8): This controls the "randomness" of the model.
   2. Low (0.0 - 0.3): The model becomes deterministic, focused, and repetitive. Good for coding or math.
   3. High (0.8 - 1.0): The model takes more risks and is more "creative." Good for story writing or brainstorming.
   4. Analogy: Low temp is a strict librarian; high temp is a wildly imaginative poet.

- Sampling (How it picks words)
   1. When the AI predicts the next word, it has a list of thousands of possibilities. Sampling filters that list.
   2. Top K (40): The AI will strictly only consider the top 40 most likely next words and ignore the thousands of other possibilities. This prevents it from saying complete nonsense.
   3. Top P (0.95): Also called "Nucleus Sampling." It cuts off the list once the combined probability of the words reaches 95%. It’s a more dynamic version of Top K.
   4. Repeat Penalty (1.1): A slight penalty applied to words the AI has already used recently. This prevents it from getting stuck in loops (e.g., "The cat sat on the the the the...").

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
ollama create knust-coe -f Modelfile
```

### Run It
```bash
ollama run grumpy-coder
```

**Task:** Ask it *"How do I center a div?"* and observe the attitude.

---



