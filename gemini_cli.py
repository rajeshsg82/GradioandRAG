#this is first version of the code. It is a simple command-line interface to interact with Google's Gemini LLM. The user can ask questions, and the script will send those questions to the Gemini API and print the responses. The API key is read from an environment variable or prompted from the user if not found.
import os
import requests

from dotenv import load_dotenv
load_dotenv()

def get_gemini_api_key():
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        api_key = input('Enter your Gemini API key: ').strip()
    return api_key

def ask_gemini(question, api_key):
    url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent'
    #url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent'
    headers = {'Content-Type': 'application/json'}
    payload = {
        'contents': [{
            'parts': [{
                'text': question
            }]
        }]
    }
    params = {'key': api_key}
    response = requests.post(url, headers=headers, params=params, json=payload)
    if response.status_code == 200:
        data = response.json()
        try:
            return data['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            return 'No answer found.'
    else:
        return f'Error: {response.status_code} - {response.text}'

def main():
    api_key = get_gemini_api_key()
    print('Ask questions to Gemini LLM (type "exit" to quit)')
    while True:
        question = input('You: ')
        if question.lower() in ('exit', 'quit'):
            print('Goodbye!')
            break
        answer = ask_gemini(question, api_key)
        print(f'Gemini: {answer}\n')

if __name__ == '__main__':
    main()
