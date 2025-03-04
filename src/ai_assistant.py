import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
import torch.nn as nn
import torch.optim as optim
import time
import random
import re
import numpy as np
from collections import Counter

# Simplification complète de l'assistant pour ne compter que sur le modèle entraîné


def get_model_paths():
    """Retourne une liste des chemins possibles pour le modèle, en privilégiant le dossier principal"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Chemin principal (celui qui doit être privilégié)
    main_model_path = os.path.join(project_dir, "models", "trained")

    # Autres chemins de fallback, dans l'ordre de préférence
    other_paths = [
        os.path.join(project_dir, "trained_llm"),
        "dbddv01/gpt2-french-small",  # Modèle par défaut si aucun modèle entraîné n'est trouvé
    ]

    # On place le chemin principal en premier
    return [main_model_path] + other_paths


def load_model(verbose=True):
    """Charge le modèle et le tokenizer depuis les chemins disponibles"""
    if verbose:
        print("⏳ Chargement du modèle et du tokenizer...")

    model_paths = get_model_paths()
    model = None
    tokenizer = None
    used_path = None

    for path in model_paths:
        if verbose:
            print(f"Tentative de chargement depuis: {path}")
        try:
            # Vérifier que c'est un dossier qui contient les fichiers nécessaires
            if os.path.isdir(path) and any(
                file.endswith(".bin") for file in os.listdir(path)
            ):
                tokenizer = AutoTokenizer.from_pretrained(path)
                model = AutoModelForCausalLM.from_pretrained(path)
                used_path = path
                break
            elif not os.path.isdir(path):  # Si c'est un modèle HuggingFace Hub
                tokenizer = AutoTokenizer.from_pretrained(path)
                model = AutoModelForCausalLM.from_pretrained(path)
                used_path = path
                break
        except (OSError, ValueError, FileNotFoundError) as e:
            if verbose:
                print(f"Échec: {str(e)}")
            continue

    # Si on n'a pas trouvé de modèle principal, alors seulement chercher dans les sous-dossiers
    if model is None:
        if verbose:
            print(
                "Aucun modèle trouvé dans le dossier principal, recherche dans les sous-dossiers..."
            )

        # Chemin principal pour chercher les best_model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        main_model_dir = os.path.join(project_dir, "models", "trained")

        if os.path.isdir(main_model_dir):
            import glob

            # Chercher les dossiers best_model_* et les trier par score
            best_model_pattern = os.path.join(main_model_dir, "best_model_*")
            best_model_paths = sorted(
                glob.glob(best_model_pattern),
                key=lambda x: float(x.split("_")[-1]),
                reverse=False,
            )  # Du meilleur (plus bas) au pire

            for path in best_model_paths:
                if verbose:
                    print(f"Tentative de chargement depuis (sous-dossier): {path}")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(path)
                    model = AutoModelForCausalLM.from_pretrained(path)
                    used_path = path
                    break
                except (OSError, ValueError) as e:
                    if verbose:
                        print(f"Échec: {str(e)}")
                    continue

    if model is None:
        raise ValueError(
            "Impossible de charger le modèle depuis les chemins disponibles. "
            "Assurez-vous que le dossier 'models/trained' contient un modèle valide."
        )

    # Déplacer le modèle sur GPU si disponible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if verbose:
        print(f"✅ Modèle chargé depuis {used_path} sur {device.upper()}")

    return model, tokenizer


class QuestionAnalyzer(nn.Module):
    """Petit réseau de neurones pour analyser les questions et identifier leur type"""

    def __init__(self, vocab_size=5000, embedding_dim=64, hidden_dim=32, output_dim=6):
        super(QuestionAnalyzer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # bidirectional -> *2
        self.softmax = nn.Softmax(dim=1)

        # Catégories de questions que le réseau peut identifier
        self.categories = [
            "factual",
            "opinion",
            "how-to",
            "description",
            "explanation",
            "other",
        ]

        # Initialisation explicite avec des biais appropriés
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialise les poids du réseau avec des biais pour les types de questions courants"""
        # Assurer une meilleure détection des questions factuelles vs opinions
        torch.nn.init.xavier_uniform_(self.fc.weight)
        # Bias factuel plus élevé pour reconnaître les questions factuelles
        self.fc.bias.data = torch.tensor(
            [0.5, -0.3, -0.2, -0.2, -0.2, -0.1], dtype=torch.float
        )

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        # Prendre uniquement la dernière sortie pour la classification
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return self.softmax(out)

    def predict(self, question, tokenizer):
        """Prédit la catégorie d'une question avec une analyse plus robuste"""
        # Normaliser la question
        question = question.lower().strip()

        # Détection directe pour les questions factuelles courantes
        factual_patterns = [
            r"quelle\s+(est|sont)",
            r"qu'est(\s+|-)+ce que",
            r"(où|quand|qui|pourquoi|comment|combien)",
            r"(capitale|superficie|population|nombre|taille|date|couleur)",
        ]

        for pattern in factual_patterns:
            if re.search(pattern, question):
                return "factual", 0.9

        # Suite du processus normal pour les cas ambigus
        words = question.split()
        # Tokenisation simplifiée
        word_to_idx = {word: min(hash(word) % 4999, 4999) for word in words}
        # Convertir la question en indices
        indices = [word_to_idx.get(word, 4999) for word in words]
        # Padding/truncation
        if len(indices) > 30:
            indices = indices[:30]
        else:
            indices = indices + [0] * (30 - len(indices))

        # Convertir en tensor et prédire
        tensor = torch.tensor([indices], dtype=torch.long)
        with torch.no_grad():
            outputs = self(tensor)
            _, predicted = torch.max(outputs, 1)

        # Retourner la catégorie et la confiance
        category = self.categories[predicted.item()]
        confidence = outputs[0][predicted.item()].item()
        return category, confidence


def create_or_load_analyzer():
    """Crée ou charge le réseau de neurones d'analyse de questions"""
    try:
        # Tenter de charger un modèle existant
        analyzer = QuestionAnalyzer()
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "question_analyzer.pt"
        )
        if os.path.exists(model_path):
            analyzer.load_state_dict(torch.load(model_path))
            print("Réseau d'analyse de questions chargé avec succès")
        else:
            print("Création d'un nouveau réseau d'analyse de questions")
            # Initialisation avec quelques exemples pré-entraînés
            pretrain_analyzer(analyzer)
        return analyzer
    except Exception as e:
        print(f"Erreur lors du chargement du réseau d'analyse: {str(e)}")
        # Créer un nouveau modèle en cas d'erreur
        return QuestionAnalyzer()


def pretrain_analyzer(analyzer):
    """Pré-entraîne le réseau avec quelques exemples représentatifs"""
    # Règles spécifiques pour l'embedding des mots clés
    factual_keywords = [
        "quelle",
        "qui",
        "quand",
        "où",
        "combien",
        "pourquoi",
        "capitale",
        "population",
        "superficie",
        "président",
        "nombre",
        "habitant",
        "est",
    ]
    opinion_keywords = [
        "penses",
        "crois",
        "avis",
        "opinion",
        "préfères",
        "meilleur",
        "pire",
    ]
    howto_keywords = [
        "comment",
        "procédure",
        "étapes",
        "faire",
        "créer",
        "développer",
        "fabriquer",
    ]

    # Créer un dictionnaire word_to_idx fictif pour les mots clés importants
    word_to_idx = {}
    for i, word in enumerate(factual_keywords):
        word_to_idx[word] = i + 10  # Commencer à 10

    for i, word in enumerate(opinion_keywords):
        word_to_idx[word] = i + 100  # Opinion words start at index 100

    for i, word in enumerate(howto_keywords):
        word_to_idx[word] = i + 200  # How-to words start at index 200

    # Initialiser les embeddings pour les mots clés
    for word, idx in word_to_idx.items():
        if idx < 100:  # Factual words
            # Biais vers la catégorie factuelle (indice 0)
            analyzer.embedding.weight.data[idx] = torch.tensor(
                [
                    0.5 if j == 20 else 0.1 * random.random()
                    for j in range(analyzer.embedding.weight.data[idx].shape[0])
                ],
                dtype=torch.float,
            )
        elif idx < 200:  # Opinion words
            # Biais vers la catégorie opinion (indice 1)
            analyzer.embedding.weight.data[idx] = torch.tensor(
                [
                    0.5 if j == 10 else 0.1 * random.random()
                    for j in range(analyzer.embedding.weight.data[idx].shape[0])
                ],
                dtype=torch.float,
            )
        else:  # How-to words
            # Biais vers la catégorie how-to (indice 2)
            analyzer.embedding.weight.data[idx] = torch.tensor(
                [
                    0.5 if j == 30 else 0.1 * random.random()
                    for j in range(analyzer.embedding.weight.data[idx].shape[0])
                ],
                dtype=torch.float,
            )


def generate_response(
    prompt,
    model,
    tokenizer,
    analyzer,
    max_length=150,  # Réduit pour des réponses plus concises
    temperature=0.3,  # Température réduite pour plus de focus
    top_p=0.85,
    repetition_penalty=1.2,
):
    """Génère une réponse basée uniquement sur le modèle entraîné"""
    # Analyser la question avec le réseau de neurones
    question_type, confidence = analyzer.predict(prompt, tokenizer)

    # Améliorer le prompt avec plus de structure et de contexte

    context = "Réponds de façon factuelle et concise à cette question. "

    prompt_text = prompt.strip()
    suffix = "\nRéponse: "

    # Construire le prompt final
    full_prompt = f"{context}{prompt_text}{suffix}"

    # Tokenize et convertir en tensor
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    # Paramètres de génération ajustés pour plus de précision
    gen_kwargs = {
        "max_length": max_length + len(inputs["input_ids"][0]),
        "temperature": temperature,
        "do_sample": True,
        "top_p": top_p,
        "top_k": 20,  # Réduit pour plus de précision
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": 3,
        "num_beams": 3,
        "early_stopping": True,
    }

    # Générer la réponse avec le modèle
    set_seed(42)  # Seed fixe pour la cohérence
    output_sequences = model.generate(**inputs, **gen_kwargs)

    # Décoder et extraire la réponse
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    try:
        # Extraire uniquement la partie réponse
        response = generated_text.split(suffix)[-1].strip()
    except:
        response = generated_text.strip()

    # Nettoyage amélioré de la réponse
    response = clean_response(response, prompt)

    return response, question_type


def clean_response(response, original_question):
    """Nettoie la réponse des artefacts avec un nettoyage plus agressif"""

    # Éliminer les références bibliographiques et citations
    response = re.sub(r"\([^)]*\)", "", response)
    response = re.sub(r"\[[^\]]*\]", "", response)
    response = re.sub(r"ISBN [0-9\-]+", "", response)

    # Éliminer les phrases avec des mots clés problématiques
    problem_keywords = [
        "Naissances",
        "Portail",
        "Catégorie:",
        "Notes",
        "Références",
        "présentation en ligne",
        "Jean-Pierre",
        "Jean-Paul",
        "coll.",
        "Presses universitaires",
    ]

    # Diviser en phrases
    sentences = re.split(r"([.!?]\s+)", response)
    cleaned_sentences = []
    current_sentence = ""

    for i in range(0, len(sentences)):
        if i % 2 == 0:  # Partie de texte
            current_sentence = sentences[i]
        else:  # Ponctuation
            current_sentence += sentences[i]

            # Vérifier si la phrase contient des mots problématiques
            if not any(keyword in current_sentence for keyword in problem_keywords):
                cleaned_sentences.append(current_sentence)

    cleaned_response = "".join(cleaned_sentences)

    # Nettoyage supplémentaire
    cleaned_response = re.sub(r"\[\d+\]", "", cleaned_response)
    cleaned_response = re.sub(r"\n+", " ", cleaned_response)
    cleaned_response = re.sub(r" +", " ", cleaned_response)

    # Si la réponse est trop courte ou vide après nettoyage, donner une réponse par défaut
    if len(cleaned_response.strip()) < 10:
        return "Je n'ai pas assez d'informations pour répondre précisément à cette question."

    return cleaned_response.strip()


def main():
    """Point d'entrée principal de l'assistant"""
    # Charger le modèle et le tokenizer
    model, tokenizer = load_model()

    # Charger le réseau d'analyse de questions
    analyzer = create_or_load_analyzer()

    # Affichage d'en-tête
    print("\n💬 Assistant IA Français")
    print("=" * 50)
    print("📌 COMMANDES:")
    print(" - Tapez votre question et appuyez sur Entrée")
    print(" - Tapez 'q' pour quitter")
    print("=" * 50)

    # Boucle d'interaction simplifiée
    while True:
        print("\n➤ ", end="")
        user_input = input().strip()

        if user_input.lower() in ["q", "quit", "exit"]:
            break

        if not user_input:
            continue

        print("\n🧠 Génération de la réponse...")
        start_time = time.time()

        try:
            response, question_type = generate_response(
                user_input,
                model,
                tokenizer,
                analyzer,
                max_length=250,
                temperature=0.85,
            )

            elapsed_time = time.time() - start_time
            print(f"\n🤖 Réponse (temps: {elapsed_time:.2f}s):")
            print(response)
            print("\n" + "=" * 50)

        except Exception as e:
            print(f"❌ Erreur lors de la génération de la réponse: {str(e)}")
            import traceback

            traceback.print_exc()

    print("\nAu revoir! 👋")


if __name__ == "__main__":
    main()
