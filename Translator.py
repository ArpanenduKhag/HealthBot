# Translator.py

from playsound import playsound
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import os
import pyttsx3
from langdetect import detect
import pycountry

class HealthBotTranslator:
    def __init__(self):

    # Initialize necessary components

    def take_command(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("listening.....")
            r.pause_threshold = 1
            audio = r.listen(source)

        try:
            print("Recognizing.....")
            query = r.recognize_google(audio, language="en-in")
            print(f"The User said {query}\n")
        except Exception as e:
            print("say that again please.....")
            return "None"
        return query# Implement speech recognition logic
        
    def get_lang_name(self, lang_code):
        anguage = pycountry.languages.get(alpha_2=lang_code)
        return language.name# Implement language code to language name conversion

    def destination_language(self):
        print("Enter the language in which you	want to convert : Ex. Hindi , English , Spanish, etc.")
        print()

        # Input destination language in
        # which the user wants to translate
        to_lang = takecommand()
        while to_lang == "None":
            to_lang = takecommand()
        to_lang = to_lang.lower()
        return to_lang# Implement logic to get the destination language

    def translate_text(self, text, from_lang, to_lang):
        translator = Translator(service_urls=['translate.googleapis.com'])
        text_to_translate=translator.translate( query , dest= to_lang )

        #text_to_translate = translator.translate(query, dest= to_lang)

        text = text_to_translate.text# Implement translation logic

    def speak_text(self, text, lang):
        # Implement text-to-speech logic

