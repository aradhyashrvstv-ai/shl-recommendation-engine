from transformers import pipeline

def build_generator():
    return pipeline("text-generation", model="google/flan-t5-base")

def generate_response(generator, query, retrieved_text):
    prompt = f"""
A hiring manager asked: {query}

Based on the following SHL assessments:
{retrieved_text}

Provide a concise professional recommendation explaining why these assessments are suitable.
"""

    response = generator(
        prompt,
        max_new_tokens=150,
        do_sample=False
    )

    return response[0]["generated_text"]