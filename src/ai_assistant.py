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


# Définir les chemins possibles pour le modèle
def get_model_paths():
    """Retourne une liste des chemins possibles pour le modèle"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    return [
        os.path.join(project_dir, "models", "trained"),
        os.path.join(
            project_dir, "models", "trained", "best_model_*"
        ),  # Pour attraper les meilleurs modèles
        os.path.join(project_dir, "trained_llm"),
        "dbddv01/gpt2-french-small",  # Modèle par défaut si aucun modèle entraîné n'est trouvé
    ]


def load_model(verbose=True):
    """Charge le modèle et le tokenizer depuis les chemins disponibles"""
    if verbose:
        print("⏳ Chargement du modèle et du tokenizer...")

    model_paths = get_model_paths()
    model = None
    tokenizer = None
    used_path = None

    import glob

    # Chercher d'abord le meilleur modèle sauvegardé (avec le score dans le nom)
    best_model_paths = glob.glob(model_paths[1])
    if best_model_paths:
        # Trier les meilleurs modèles par score (supposant que le score est dans le nom)
        # Format attendu: best_model_3.1234
        best_model_paths.sort(key=lambda x: float(x.split("_")[-1]))
        model_paths.insert(
            0, best_model_paths[0]
        )  # Ajouter le meilleur modèle au début de la liste

    for path in model_paths:
        if verbose:
            print(f"Tentative de chargement depuis: {path}")
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
            "Impossible de charger le modèle depuis les chemins disponibles"
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

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        # Prendre uniquement la dernière sortie pour la classification
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return self.softmax(out)

    def predict(self, question, tokenizer):
        """Prédit la catégorie d'une question"""
        # Créer un vocabulaire simple à partir des tokens du modèle
        words = question.lower().split()
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
    # Simule un petit entraînement avec des exemples représentatifs
    # En pratique, on utiliserait un dataset plus large et une vraie phase d'entraînement

    # Pour la simplicité de la démonstration, on initialise juste quelques poids
    # qui favorisent la détection de mots clés spécifiques à chaque catégorie

    # Initialisation des poids de l'embedding pour des mots clés
    factual_keywords = [
        "quelle",
        "qui",
        "quand",
        "où",
        "combien",
        "pourquoi",
        "est-ce",
        "capitale",
    ]
    opinion_keywords = ["penses-tu", "crois-tu", "avis", "opinion", "préfères"]
    howto_keywords = ["comment", "procédure", "étapes", "faire", "créer", "développer"]

    # Initialiser quelques poids pour biaiser légèrement le modèle vers ces catégories
    # Ceci est une simplification - normalement on entraînerait un vrai modèle
    for i in range(analyzer.embedding.weight.shape[0]):
        analyzer.embedding.weight.data[i] = (
            torch.randn(analyzer.embedding.weight.data[i].shape) * 0.1
        )


def generate_response(
    prompt,
    model,
    tokenizer,
    analyzer,
    max_length=150,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.2,
):
    """Génère une réponse basée sur le prompt donné, en utilisant l'analyseur de questions"""
    # Analyser la question pour déterminer son type
    question_type, confidence = analyzer.predict(prompt, tokenizer)

    # Ajuster les paramètres de génération en fonction du type de question
    if question_type == "factual":
        # Pour les questions factuelles, réduire la température pour des réponses plus précises
        temperature *= 0.6
        max_length = min(max_length, 100)  # Réponses plus courtes et précises
        prefix = "Question factuelle: "
        suffix = "\nRéponse précise et factuelle: "
    elif question_type == "opinion":
        # Pour les questions d'opinion, permettre plus de créativité
        temperature *= 1.1
        prefix = "Question d'opinion: "
        suffix = "\nRéponse exprimant une opinion: "
    elif question_type == "how-to":
        # Pour les instructions, être méthodique
        temperature *= 0.8
        max_length = max(
            max_length, 200
        )  # Permettre des réponses plus longues pour les instructions
        prefix = "Question sur une procédure: "
        suffix = "\nRéponse détaillant les étapes: "
    else:
        # Par défaut
        prefix = "Question: "
        suffix = "\nRéponse: "

    # Construire le prompt complet avec instructions pour le modèle
    context = "Tu es un assistant IA français qui répond de manière claire et précise. "

    # Ajouter des instructions spécifiques selon le type de question
    if question_type == "factual":
        context += "Cette question demande des faits précis. Réponds de façon concise et factuelle. "
    elif question_type == "opinion":
        context += (
            "Cette question demande une opinion. Exprime un point de vue équilibré. "
        )
    elif question_type == "how-to":
        context += "Cette question demande une procédure. Explique les étapes de façon claire. "

    full_prompt = f"{context}{prefix}{prompt.strip()}{suffix}"

    # Tokenize et convertir en tensor
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    # Configurer les paramètres de génération
    gen_kwargs = {
        "max_length": max_length + len(inputs["input_ids"][0]),
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "do_sample": True,
        "top_k": 30,
        "no_repeat_ngram_size": 3,
        "num_beams": 5 if question_type == "factual" else 3,
        "early_stopping": True,
    }

    # Générer la réponse
    set_seed(random.randint(1, 1000))
    output_sequences = model.generate(**inputs, **gen_kwargs)

    # Décoder et nettoyer la réponse
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    response = generated_text.split(suffix)[-1].strip()

    # Nettoyer la réponse
    response = clean_response(response)

    return response, question_type


# def check_factual_question(question):
#     """Vérifie si la question est factuelle simple et retourne une réponse pré-définie si applicable"""
#     # Normalisation de la question
#     q = question.lower().strip()

#     # Liste de réponses factuelles pour les questions courantes
#     factual_answers = {
#         "capitale de la france": "Paris est la capitale de la France.",
#         "capital de la france": "Paris est la capitale de la France.",
#         "quelle est la capitale de la france": "Paris est la capitale de la France.",
#         "population de la france": "La France a environ 67 millions d'habitants (estimation 2023).",
#         "président de la france": "Emmanuel Macron est le président de la République française depuis 2017.",
#         "capitale de l'italie": "Rome est la capitale de l'Italie.",
#         "capitale de l'allemagne": "Berlin est la capitale de l'Allemagne.",
#         "capitale de l'espagne": "Madrid est la capitale de l'Espagne.",
#         "capitale du royaume-uni": "Londres est la capitale du Royaume-Uni.",
#         "capitale des états-unis": "Washington D.C. est la capitale des États-Unis.",
#         "capitale du canada": "Ottawa est la capitale du Canada.",
#         "capitale du japon": "Tokyo est la capitale du Japon.",
#         "capitale de la chine": "Pékin (Beijing) est la capitale de la Chine.",
#         "capitale de la russie": "Moscou est la capitale de la Russie.",
#     }

#     # Vérifier si la question correspond à une entrée dans notre dictionnaire
#     for key, answer in factual_answers.items():
#         if key in q or q in key:
#             return answer

#     # Pour les questions de capitale qui suivent un format standard
#     capital_match = re.search(
#         r"capitale\s+(du|de la|de l'|des)\s+([a-zÀ-ÿ\s]+)(\s*\?)?", q
#     )
#     if capital_match:
#         country = capital_match.group(2).strip()
#         # Ajouter ici une liste plus complète de pays et leurs capitales
#         capitals = {
#             "portugal": "Lisbonne est la capitale du Portugal.",
#             "suisse": "Berne est la capitale de la Suisse.",
#             "belgique": "Bruxelles est la capitale de la Belgique.",
#             "pays-bas": "Amsterdam est la capitale des Pays-Bas.",
#             "australie": "Canberra est la capitale de l'Australie.",
#             "brésil": "Brasilia est la capitale du Brésil.",
#             "mexique": "Mexico est la capitale du Mexique.",
#             "argentine": "Buenos Aires est la capitale de l'Argentine.",
#             "inde": "New Delhi est la capitale de l'Inde.",
#             "corée du sud": "Séoul est la capitale de la Corée du Sud.",
#             "égypte": "Le Caire est la capitale de l'Égypte.",
#             "afrique du sud": "Pretoria est la capitale administrative de l'Afrique du Sud.",
#         }
#         if country in capitals:
#             return capitals[country]

#     return None


def clean_response(response):
    """Nettoie la réponse des artefacts courants et améliore sa lisibilité"""
    # Éliminer les phrases qui contiennent des mots clés problématiques
    problem_keywords = [
        "Naissances",
        "Portail",
        "Catégorie:",
        "Le Monde",
        "Notes",
        "Références",
        "Articles connexes",
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

    # Nettoyer les références Wikipedia, hashtags, etc.
    cleaned_response = re.sub(r"#[a-zA-Z0-9_]+", "", cleaned_response)
    cleaned_response = re.sub(r"\[\d+\]", "", cleaned_response)
    cleaned_response = re.sub(r"\(lire en ligne\)", "", cleaned_response)

    # Enlever les lignes qui commencent par des caractères spéciaux ou des statistiques
    cleaned_response = re.sub(
        r"^\s*[#\*\-][^\n]*$", "", cleaned_response, flags=re.MULTILINE
    )

    # Éliminer les lignes trop courtes (souvent des artefacts)
    cleaned_response = re.sub(r"^\s*.{1,10}$", "", cleaned_response, flags=re.MULTILINE)

    # Éliminer les sauts de ligne multiples et espaces en trop
    cleaned_response = re.sub(r"\n+", "\n", cleaned_response)
    cleaned_response = re.sub(r" +", " ", cleaned_response)

    # Si après nettoyage la réponse est trop courte, renvoyez un message d'excuse
    if len(cleaned_response.strip()) < 10:
        return "Je ne peux pas fournir une réponse précise à cette question avec les informations dont je dispose."

    return cleaned_response.strip()


def main():
    """Point d'entrée principal de l'assistant"""
    # Charger le modèle et le tokenizer
    model, tokenizer = load_model()

    # Charger le réseau d'analyse de questions
    analyzer = create_or_load_analyzer()

    # Afficher l'en-tête
    print("\n💬 Assistant IA Français avec réseau neuronal d'analyse")
    print("=" * 50)
    print("📌 COMMANDES:")
    print(" - Tapez votre question et appuyez sur Entrée")
    print(" - Tapez 'q' pour quitter")
    print("=" * 50)

    # Boucle d'interaction
    while True:
        print("\n➤ ", end="")
        user_input = input().strip()

        if user_input.lower() in ["q", "quit", "exit"]:
            break

        if not user_input:
            continue

        print("\n🧠 Réflexion en cours...")
        start_time = time.time()

        try:
            # Générer la réponse avec l'aide du réseau d'analyse
            response, question_type = generate_response(
                user_input, model, tokenizer, analyzer, max_length=150, temperature=0.7
            )

            # Si la réponse semble problématique, faire une seconde tentative
            if len(response) < 15 or any(
                word in response for word in ["Naissances", "Portail", "Catégorie"]
            ):
                print("(Amélioration de la réponse...)")
                response, _ = generate_response(
                    user_input,
                    model,
                    tokenizer,
                    analyzer,
                    max_length=100,
                    temperature=0.5,
                    repetition_penalty=1.5,
                )

            elapsed_time = time.time() - start_time

            print(f"\n🤖 Réponse (type: {question_type}, temps: {elapsed_time:.2f}s):")
            print(response)
            print("\n" + "=" * 50)

            # Apprentissage continu du réseau d'analyse
            # En production, on utiliserait un mécanisme pour recueillir le feedback utilisateur
            # et mettre à jour périodiquement le modèle d'analyse

        except Exception as e:
            print(f"❌ Erreur lors de la génération de la réponse: {str(e)}")

    print("\nAu revoir! 👋")

    # Sauvegarder les améliorations du réseau d'analyse (en production)
    # torch.save(analyzer.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "question_analyzer.pt"))


if __name__ == "__main__":
    main()
