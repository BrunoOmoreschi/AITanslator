"""Tradutor de frases em vários idiomas. Funcionando! Usa o modelo M2M100-418."""
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Função para exibir o menu de opções
def show_menu():
    print("\nMenu de Opções:")
    print("1. Inserir texto a ser traduzido")
    print("2. Selecionar idioma de origem")
    print("3. Selecionar idioma de destino")
    print("4. Exibir lista de códigos de idiomas suportados")
    print("5. Traduzir texto")
    print("0. Sair")

# Função para exibir a lista de códigos de idiomas suportados
def show_language_codes():
    print("\nLista de Códigos de Idiomas Suportados:")
    # Lista dos códigos de idiomas suportados pelo modelo M2M100
    language_codes = [
        'zh - Chinês',
        'en - Inglês',
        'hi - Hindi',
        'es - Espanhol',
        'ar - Árabe',
        'bn - Bengali',
        'pt - Português',
        'ru - Russo',
        'ja - Japonês',
        'pa - Punjabi',
        'de - Alemão',
        'jv - Javanês',
        'te - Telugu',
        'vi - Vietnamita',
        'ko - Coreano',
        'fr - Francês',
        'mr - Marathi',
        'ta - Tâmil',
        'ur - Urdu',
        'tr - Turco',
        'it - Italiano',
        'yo - Iorubá',
        'th - Tailandês',
        'gu - Gujarati',
        'fa - Persa',
        'ml - Malaiala',
        'uk - Ucraniano',
        'ro - Romeno',
        'nl - Holandês',
        'sw - Swahili',
        'am - Amárico',
        'my - Birmanês',
        'sd - Sindhi',
        'sn - Shona',
        'ps - Pashto',
        'lt - Lituano',
        'ku - Curdo',
        'ms - Malaio',
        'az - Azerbaijano',
        'bg - Búlgaro',
        'sr - Sérvio',
        'ka - Georgiano',
        'hr - Croata',
        'kk - Cazaque',
        'el - Grego',
        'hu - Húngaro',
        'vi - Vietnamita',
        'kn - Kannada',
        'ny - Chichewa',
        'ms - Malaio',
        'ceb - Cebuano'
    ]
    # Esses são os códigos de idiomas que ele faz traduções razoaveis, segundo o ChatGpt (15/7/24).

    for code in language_codes:
        print(code)

# Inicializando variáveis
text = ""
source_lang = ""
target_lang = ""

# Carregar o modelo e o tokenizer
model_name = "facebook/m2m100_418M"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Loop principal do menu
while True:
    show_menu()
    choice = input("\nEscolha uma opção: ")

    if choice == '1':
        text = input("\nDigite o texto a ser traduzido: ")
    elif choice == '2':
        source_lang = input("\nDigite o código do idioma de origem: ")
    elif choice == '3':
        target_lang = input("\nDigite o código do idioma de destino: ")
    elif choice == '4':
        show_language_codes()
    elif choice == '5':
        if not text:
            print("\nTexto não fornecido. Por favor, insira o texto a ser traduzido.")
        elif not source_lang or not target_lang:
            print("\nCódigo de idioma não fornecido. Por favor, insira os códigos dos idiomas de origem e destino.")
        else:
            # Tradução
            inputs = tokenizer(text, return_tensors="pt")
            generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
            result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            print(f"\nTexto traduzido ({source_lang} -> {target_lang}): {result}")
    elif choice == '0':
        print("\nSaindo...")
        break
    else:
        print("\nOpção inválida. Por favor, escolha uma opção válida.")
