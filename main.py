'''This is a file containing the solution created by the chatGPT with the same prompt as furnished to the local AI,
the phi3.5. Turns out that was necessary give more information about the language list and the model used to
tranlate as the code that was used ins the m2mLoader.py file.'''

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.spinner import Spinner
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.button import Button
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class TranslatorApp(App):
    languages = [
        'zh - Chinês', 'en - Inglês', 'hi - Hindi', 'es - Espanhol', 'ar - Árabe', 'bn - Bengali',
        'pt - Português', 'ru - Russo', 'ja - Japonês', 'pa - Punjabi', 'de - Alemão', 'jv - Javanês',
        'te - Telugu', 'vi - Vietnamita', 'ko - Coreano', 'fr - Francês', 'mr - Marathi', 'ta - Tâmil',
        'ur - Urdu', 'tr - Turco', 'it - Italiano', 'yo - Iorubá', 'th - Tailandês', 'gu - Gujarati',
        'fa - Persa', 'ml - Malaiala', 'uk - Ucraniano', 'ro - Romeno', 'nl - Holandês', 'sw - Swahili',
        'am - Amárico', 'my - Birmanês', 'sd - Sindhi', 'sn - Shona', 'ps - Pashto', 'lt - Lituano',
        'ku - Curdo', 'ms - Malaio', 'az - Azerbaijano', 'bg - Búlgaro', 'sr - Sérvio', 'ka - Georgiano',
        'hr - Croata', 'kk - Cazaque', 'el - Grego', 'hu - Húngaro', 'vi - Vietnamita', 'kn - Kannada',
        'ny - Chichewa', 'ceb - Cebuano'
    ]

    def build(self):
        return TranslatorLayout()

    def translate_text(self):
        text_input = self.root.ids.text_input.text
        input_lang = self.root.ids.input_lang.text.split(' - ')[0]  # Get language code
        output_lang = self.root.ids.output_lang.text.split(' - ')[0]  # Get language code

        # Load the model and tokenizer

        #Load the model with internet connection
        #model_name = "facebook/m2m100_418M"

        #Uses a previously downloaded model.
        model_name = "D:/AI/LLMs or SLMs/translationModel/models--facebook--m2m100_418M/snapshots/55c2e61bbf05dfb8d7abccdc3fae6fc8512fd636"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        inputs = tokenizer(text_input, return_tensors="pt")
        generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(output_lang))
        result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        self.root.ids.translated_text.text = f"{result}"


class TranslatorLayout(GridLayout):
    pass


if __name__ == "__main__":
    TranslatorApp().run()
