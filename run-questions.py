
from transformers import pipeline

qa = pipeline("text2text-generation", model="google/flan-t5-small")
question  = "When did Ghana gain independence?"
knowledge = """
    Ghana is a country located in West Africa. It gained independence from British colonial rule on March 6, 1957, 
    making it the first African country south of the Sahara to achieve independence. The capital city of Ghana is Accra,
    which is also the largest city in the country. Ghana was formerly known as the Gold Coast during the colonial period.
"""
result = qa("Context: " + knowledge + " Question: " + question)

print(result)