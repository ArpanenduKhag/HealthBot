from flask import Flask, render_template, request, jsonify
import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import numpy as np
import speech_recognition as sr
import pyttsx3
from googletrans import Translator as GoogleTranslator

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.h5")


class HealthBotTranslator:
    def __init__(self):
        self.engine = pyttsx3.init("sapi5")
        self.voices = self.engine.getProperty("voices")
        self.engine.setProperty("voice", self.voices[1].id)

        self.all_languages = (
            # Add your list of languages here
            "afrikaans",
            "af",
            "albanian",
            "sq",
    "amharic",
    "am",
    "arabic",
            "ar",
    "armenian",
    "hy",
    "azerbaijani",
    "az",
    "basque",
    "eu",
    "belarusian",
    "be",
    "bengali",
    "bn",
    "bosnian",
    "bs",
    "bulgarian",
    "bg",
    "catalan",
    "ca",
    "cebuano",
    "ceb",
    "chichewa",
    "ny",
    "chinese (simplified)",
    "zh-cn",
    "chinese (traditional)",
    "zh-tw",
    "corsican",
    "co",
    "croatian",
    "hr",
    "czech",
    "cs",
    "danish",
    "da",
    "dutch",
    "nl",
    "english",
    "en",
    "esperanto",
    "eo",
    "estonian",
    "et",
    "filipino",
    "tl",
    "finnish",
    "fi",
    "french",
    "fr",
    "frisian",
    "fy",
    "galician",
    "gl",
    "georgian",
    "ka",
    "german",
    "de",
    "greek",
    "el",
    "gujarati",
    "gu",
    "haitian creole",
    "ht",
    "hausa",
    "ha",
    "hawaiian",
    "haw",
    "hebrew",
    "he",
    "hindi",
    "hi",
    "hmong",
    "hmn",
    "hungarian",
    "hu",
    "icelandic",
    "is",
    "igbo",
    "ig",
    "indonesian",
    "id",
    "irish",
    "ga",
    "italian",
    "it",
    "japanese",
    "ja",
    "javanese",
    "jw",
    "kannada",
    "kn",
    "kazakh",
    "kk",
    "khmer",
    "km",
    "korean",
    "ko",
    "kurdish (kurmanji)",
    "ku",
    "kyrgyz",
    "ky",
    "lao",
    "lo",
    "latin",
    "la",
    "latvian",
    "lv",
    "lithuanian",
    "lt",
    "luxembourgish",
    "lb",
    "macedonian",
    "mk",
    "malagasy",
    "mg",
    "malay",
    "ms",
    "malayalam",
    "ml",
    "maltese",
    "mt",
    "maori",
    "mi",
    "marathi",
    "mr",
    "mongolian",
    "mn",
    "myanmar (burmese)",
    "my",
    "nepali",
    "ne",
    "norwegian",
    "no",
    "odia",
    "or",
    "pashto",
    "ps",
    "persian",
    "fa",
    "polish",
    "pl",
    "portuguese",
    "pt",
    "punjabi",
    "pa",
    "romanian",
    "ro",
    "russian",
    "ru",
    "samoan",
    "sm",
    "scots gaelic",
    "gd",
    "serbian",
    "sr",
    "sesotho",
    "st",
    "shona",
    "sn",
    "sindhi",
    "sd",
    "sinhala",
    "si",
    "slovak",
    "sk",
    "slovenian",
    "sl",
    "somali",
    "so",
    "spanish",
    "es",
    "sundanese",
    "su",
    "swahili",
    "sw",
    "swedish",
    "sv",
    "tajik",
    "tg",
    "tamil",
    "ta",
    "telugu",
    "te",
    "thai",
    "th",
    "turkish",
    "tr",
    "ukrainian",
    "uk",
    "urdu",
    "ur",
    "uyghur",
    "ug",
    "uzbek",
    "uz",
    "vietnamese",
    "vi",
    "welsh",
    "cy",
    "xhosa",
    "xh",
    "yiddish",
    "yi",
    "yoruba",
    "yo",
    "zulu",
    "zu",
        )

    def speak_text(self, audio, lang=None):
        self.engine.say(audio)
        self.engine.runAndWait()

    def get_lang_name(self, lang_code):
        language = pycountry.languages.get(alpha_2=lang_code)
        return language.name        # Implement the language name retrieval logic
        pass

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
        return query       # Implement the logic to capture user input through speech
        pass

    def detect_language(self, sentence):
        return detect(text)
       # Implement language detection logic
        pass

    def destination_language(self):
        print("Enter the language in which you want to convert: Ex. Hindi, English, Spanish, etc.")
        print()
        to_lang = self.take_command()
        while to_lang == "None":
            to_lang = self.take_command()
        to_lang = to_lang.lower()
        return to_lang
       # Implement logic for the user to choose the destination language
        pass

    def translate_text(self, sentence, from_lang, to_lang):
        # Implement translation logic
        pass
    def translate_text(self, text, from_lang, to_lang):
        translator = Translator(service_urls=['translate.googleapis.com'])
        translated_text = translator.translate(text, src=from_lang, dest=to_lang).text
        return translated_text


translator = HealthBotTranslator()


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]
    result = ""
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form["user_input"]
    predict = predict_class(user_input)
    response = get_response(predict, intents)
    return response


@app.route("/translate", methods=["POST"])
def translate():
    try:
        data = request.get_json()
        original_text = data.get("text")

        # Implement translation logic here
        translated_text = translator.translate_text(original_text, 'en', 'es')

        return jsonify({"translated_text": translated_text})
    except Exception as e:
        return jsonify({"error": str(e)})
# ... (previous code)

@app.route("/test_translation")
def test_translation():
    return render_template("test_translation.html")

# ... (rest of the code)


if __name__ == "__main__":
    app.run(debug=True)
