from transformers import pipeline

qa = pipeline("text2text-generation", model="google/flan-t5-small")

print("=== VERIFICATION WITH GHANA QUESTIONS ===")
print()

# Ghana knowledge base
ghana_knowledge = """
    Ghana is a country located in West Africa. It gained independence from British colonial rule on March 6, 1957, 
    making it the first African country south of the Sahara to achieve independence. The capital city of Ghana is Accra,
    which is also the largest city in the country. Ghana was formerly known as the Gold Coast during the colonial period.
"""

# Test different questions about Ghana
questions = [
    ("When did Ghana gain independence?", "1957"),
    ("What is the capital of Ghana?", "Accra"),
    ("What was Ghana formerly called?", "Gold Coast")
]

for i, (question, expected_answer) in enumerate(questions, 1):
    print(f"=== QUESTION {i}: {question} ===")
    
    # Test 1: With knowledge context
    result1 = qa("Context: " + ghana_knowledge + " Question: " + question)
    print("WITH KNOWLEDGE CONTEXT:")
    print(f"   Answer: {result1[0]['generated_text']}")
    print(f"   Correct: {'✓' if expected_answer.lower() in result1[0]['generated_text'].lower() else '✗'}")
    print()
    
    # Test 2: Without knowledge context
    result2 = qa("Context:  Question: " + question)
    print("WITHOUT KNOWLEDGE CONTEXT:")
    print(f"   Answer: {result2[0]['generated_text']}")
    print(f"   Correct: {'✓' if expected_answer.lower() in result2[0]['generated_text'].lower() else '✗'}")
    print()
    print("-" * 50)

print("=== CONCLUSION ===")
print("This demonstrates how context-dependent language models work:")
print("- With proper context: Models can answer questions accurately")
print("- Without context: Models often give incorrect or irrelevant answers")
print("- Context is crucial for reliable question-answering systems")