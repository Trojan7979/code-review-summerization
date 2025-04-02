from groq import Groq
import os
import re

def message_response(msg, token, model):
    client = Groq(api_key=token)
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a specialist in code analysis."},
            {"role": "user", "content": msg}],
        temperature=0.6,
        max_completion_tokens=4096,
        top_p=0.95
    )

    # for chunk in completion:
    #     print(chunk.choices[0].delta.content or "", end="")
    res = remove_think_tags(completion.choices[0].message.content)
    return res

def speech_to_text_llm(audio_path, token):
    client = Groq(api_key=token)
    with open(audio_path, "rb") as file:

        # Create a translation of the audio file
        translation = client.audio.translations.create(
        file=(audio_path, file.read()),
        model="whisper-large-v3",
        prompt="Specify context or spelling",
        response_format="json",
        temperature=0.0
        )
        print(translation.text)
    return translation.text

def remove_think_tags(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

if __name__ == "__main__":
    print(message_response("Hi"))