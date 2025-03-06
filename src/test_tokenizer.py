from tokenizers import Tokenizer
import os
import argparse

# 🔍 Déterminer le chemin racine du projet
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "data", "tokenized", "tokenizer.json")


def test_tokenization(text, show_ids=False, show_stats=False):
    """Tokenise un texte et affiche les résultats"""
    try:
        # Charger le tokenizer
        if not os.path.exists(TOKENIZER_PATH):
            print(f"❌ Erreur: Le fichier tokenizer '{TOKENIZER_PATH}' n'existe pas.")
            print("   Exécutez d'abord 'train_tokenizer.py' pour générer le tokenizer.")
            return

        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        print(f"✅ Tokenizer chargé depuis {TOKENIZER_PATH}")

        # Tokeniser le texte
        print(f'\n📝 Texte original: "{text}"\n')
        encoding = tokenizer.encode(text)

        # Afficher les tokens
        if show_ids:
            print(f"🔢 Tokens avec IDs:")
            for i, (token, id) in enumerate(zip(encoding.tokens, encoding.ids)):
                print(f'  {i+1:>3}. "{token}" (ID: {id})')
        else:
            print(f"🔤 Tokens: {encoding.tokens}")

        # Afficher des statistiques
        if show_stats:
            print("\n📊 Statistiques:")
            print(f"  - Nombre de tokens: {len(encoding.tokens)}")
            vocab_size = tokenizer.get_vocab_size()
            print(f"  - Taille du vocabulaire: {vocab_size} tokens")
            print(
                f"  - Ratio de compression: {len(text)/len(encoding.tokens):.2f} caractères/token"
            )

    except Exception as e:
        print(f"❌ Erreur lors de la tokenisation: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test du tokenizer préentraîné")
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default="Bonjour, je suis un exemple de texte pour tester le tokenizer.",
        help="Texte à tokeniser",
    )
    parser.add_argument(
        "--ids", "-i", action="store_true", help="Afficher les IDs des tokens"
    )
    parser.add_argument(
        "--stats",
        "-s",
        action="store_true",
        help="Afficher des statistiques sur la tokenisation",
    )

    args = parser.parse_args()
    test_tokenization(args.text, args.ids, args.stats)

    # Mode interactif si aucun texte n'est spécifié par ligne de commande
    if not args.text and os.isatty(0):
        print("\n🔄 Mode interactif: tapez 'exit' pour quitter")
        while True:
            text = input("\n🖊️  Entrez un texte à tokeniser: ")
            if text.lower() == "exit":
                break
            test_tokenization(text, args.ids, args.stats)
