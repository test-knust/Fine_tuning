# 🇬🇭 Ghana Knowledge Fine-Tuning Project

**Parameter Efficient Fine-Tuning (PEFT) with LoRA for Domain-Specific Knowledge**

This repository demonstrates successful fine-tuning of FLAN-T5-small on Ghana-specific knowledge using LoRA (Low-Rank Adaptation), achieving **66.7% accuracy improvement** over the base model.

## 🎯 **Project Highlights**

- ✅ **Successful Fine-tuning**: 66.7% accuracy vs 0% baseline
- ⚡ **Efficient Training**: Only 1.76% of parameters trained
- 🚀 **Fast Training**: ~48 seconds for 10 epochs
- 📊 **Real Results**: March 6, 1957 vs random years like 1897
- 🎓 **Educational**: Perfect for learning tokenization and PEFT

## 📁 **Project Structure**

```
Fine_tuning/
├── 📊 ghana_qa.json              # Training data (111 Q&A pairs)
├── 🤖 finetune_ghana.py          # Working fine-tuning script  
├── 🧪 test_finetuned_model.py    # Model testing and comparison
├── 🔍 run-questions.py           # Original context-based approach
├── ✅ verify-statement.py        # Context vs no-context demo
├── 📋 requirements.txt           # Python dependencies
├── 🔧 verify_setup.py            # Environment verification
└── 📝 README.md                  # This file
```

## 🚀 **Quick Start**

### Prerequisites
- **Python 3.10+** is required
- **VS Code** (recommended): https://code.visualstudio.com

### 1. Setup Environment

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```batch
python -m venv .venv
.venv\Scripts\activate.bat
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Setup

```bash
python verify_setup.py
```

Expected Output:
```
Torch version: 2.x.x
Transformers version: 4.x.x
CUDA available: True/False
```

### 4. Run the Project

**Option A: Fine-tune the model**
```bash
python3 finetune_ghana.py
```

**Option B: Test existing approach**
```bash
python3 run-questions.py              # Context-based approach
python3 verify-statement.py           # Compare context vs no-context
```

**Option C: Test fine-tuned model** (after fine-tuning)
```bash
python3 test_finetuned_model.py
```

## 📊 **Results Demonstration**

### **Before Fine-tuning (Context Required):**
```python
# With context
qa("Context: Ghana gained independence in 1957... Question: When did Ghana gain independence?")
# Answer: "March 6, 1957" ✅

# Without context  
qa("When did Ghana gain independence?")
# Answer: "1897" ❌ (random/wrong)
```

### **After Fine-tuning (No Context Needed):**
```python
# Direct question - knowledge embedded in model
qa("When did Ghana gain independence?")
# Answer: "March 6, 1957" ✅

qa("What is the capital of Ghana?")  
# Answer: "Accra" ✅

qa("Where is Ghana located?")
# Answer: "West Africa" ✅
```

## 🎓 **Educational Value**

### **Key Learning Concepts:**
1. **🔤 Tokenization**: Text ↔ Numbers conversion
2. **📏 Padding/Truncation**: Batch processing techniques  
3. **🎯 PEFT/LoRA**: Efficient fine-tuning methods
4. **📊 Training Data**: Preparation and formatting
5. **🔍 Evaluation**: Model comparison and metrics

### **Technical Achievements:**
- **Training Data**: 111 Ghana Q&A pairs → focused 20 examples
- **Model**: FLAN-T5-small with LoRA adapters
- **Parameters**: Only 1,376,256 trainable (1.76% of total)
- **Training Time**: 48 seconds for 10 epochs
- **Loss Reduction**: 3.47 → 0.27 (excellent learning curve)

## 🔬 **Technical Details**

### **LoRA Configuration:**
```python
LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,                          # Rank (complexity)
    lora_alpha=32,                 # Scaling factor
    lora_dropout=0.1,              # Regularization
    target_modules=["q", "v", "k", "o"]  # Attention layers
)
```

### **Training Parameters:**
- **Learning Rate**: 1e-3 (higher than typical)
- **Epochs**: 10 (focused training)
- **Batch Size**: 2 (memory efficient)
- **Max Length**: 128 tokens

## 📈 **Performance Metrics**

| Model | Accuracy | Example Answer |
|-------|----------|----------------|
| **Original (no context)** | 0/6 (0.0%) | "1897" for independence |
| **Fine-tuned** | 4/6 (66.7%) | "March 6, 1957" ✅ |
| **Improvement** | **+66.7%** | Real knowledge embedded |

### **Successful Answers:**
- ✅ Independence date: "March 6, 1957"
- ✅ Capital city: "Accra"  
- ✅ Location: "West Africa"
- ✅ Independence year: "1957"

## 🛠️ **Dependencies**

Core libraries installed via `requirements.txt`:
- **torch** - Deep learning framework
- **transformers** - Hugging Face model library  
- **datasets** - Data processing utilities
- **peft** - Parameter Efficient Fine-Tuning
- **accelerate** - Training acceleration
- **numpy, pandas** - Data manipulation

## 🎯 **Use Cases**

### **Educational:**
- Learn tokenization and data preparation
- Understand PEFT and LoRA techniques
- Practice model evaluation and comparison
- Explore domain-specific knowledge embedding

### **Real-World Applications:**
- **Customer Support**: Company-specific FAQ bots
- **Medical Q&A**: Specialized medical knowledge
- **Legal Assistant**: Domain-specific legal information  
- **Product Knowledge**: Embed product catalogs in models

## 🚧 **Next Steps**

1. **Expand Dataset**: Add more Ghana questions (economy, culture, geography)
2. **Try Different Models**: Test with FLAN-T5-base or other architectures
3. **Experiment with LoRA**: Adjust rank, alpha, target modules
4. **Add Evaluation**: Implement BLEU, ROUGE scoring
5. **Deploy Model**: Create inference API or web interface

## 🤝 **Contributing**

This project is perfect for learning and experimentation! Try:
- Adding questions about other countries
- Experimenting with different LoRA configurations  
- Testing other model architectures
- Implementing additional evaluation metrics

## 📞 **Support**

Run `python verify_setup.py` to diagnose environment issues.

Common solutions:
- **Import errors**: Check virtual environment activation
- **CUDA issues**: Verify PyTorch installation  
- **Memory errors**: Reduce batch size or model size

---

**🎉 This project demonstrates successful domain-specific fine-tuning with measurable results - perfect for learning modern NLP techniques!**
