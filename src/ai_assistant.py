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

# Dictionnaire de réponses prédéfinies pour les questions les plus courantes
FACTUAL_ANSWERS = {
    "capitale france": "La capitale de la France est Paris.",
    "capitale italie": "La capitale de l'Italie est Rome.",
    "capitale allemagne": "La capitale de l'Allemagne est Berlin.",
    "capitale espagne": "La capitale de l'Espagne est Madrid.",
    "capitale royaume-uni": "La capitale du Royaume-Uni est Londres.",
    "capitale angleterre": "Londres est la capitale de l'Angleterre.",
    "capitale etats-unis": "La capitale des États-Unis est Washington D.C.",
    "capitale usa": "La capitale des États-Unis est Washington D.C.",
    "capitale canada": "La capitale du Canada est Ottawa.",
    "capitale japon": "La capitale du Japon est Tokyo.",
    "capitale chine": "La capitale de la Chine est Pékin (Beijing).",
    "capitale russie": "La capitale de la Russie est Moscou.",
    "population france": "La France compte environ 67 millions d'habitants (estimation 2023).",
    "president france": "Le président de la République française est Emmanuel Macron depuis 2017.",
    "langue france": "La langue officielle de la France est le français.",
    "monnaie france": "La monnaie de la France est l'euro (€).",
}


def get_factual_answer(question):
    """Vérifie si la question correspond à une réponse factuelle prédéfinie"""
    # Normaliser la question: retirer ponctuation et mettre en minuscules
    normalized = question.lower()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # Recherche exacte
    for key, answer in FACTUAL_ANSWERS.items():
        search_patterns = [
            f"quelle est la {key}",
            f"quel est le {key}",
            f"qui est le {key}",
            f"{key} est",
            f"{key}",
        ]
        if any(pattern in normalized for pattern in search_patterns):
            return answer

    # Recherche plus complexe pour les capitales
    if "capitale" in normalized:
        for key, answer in FACTUAL_ANSWERS.items():
            if "capitale" in key and key.split()[1] in normalized:
                return answer

    return None


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
    max_length=150,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.2,
):
    """Génère une réponse basée sur le prompt donné, en utilisant l'analyseur de questions"""
    # D'abord essayer de trouver une réponse dans les réponses prédéfinies
    factual_answer = get_factual_answer(prompt)
    if factual_answer:
        return factual_answer, "factual"

    # Si pas de correspondance, analyser la question pour déterminer son type
    question_type, confidence = analyzer.predict(prompt, tokenizer)

    # Construire un prompt approprié selon le type de question
    if question_type == "factual":
        # Pour les questions factuelles, utiliser un prompt très directif
        temperature = 0.3  # Température très basse pour des réponses plus déterministes
        max_length = 50  # Limiter la longueur pour éviter les divagations

        # Construire un prompt spécifique aux faits
        context = "Réponds à cette question factuelle de manière concise et directe. "
        prefix = "Question: "
        suffix = "\nRéponse factuelle: "

        # Pour certains types spécifiques de questions, donner des indices
        if "capitale" in prompt.lower():
            context += (
                "Les capitales des pays sont des informations factuelles précises. "
            )
        elif "population" in prompt.lower() or "habitant" in prompt.lower():
            context += (
                "Les données démographiques sont des informations factuelles précises. "
            )

    elif question_type == "opinion":
        # Pour les questions d'opinion, permettre plus de créativité
        temperature = 0.8
        context = (
            "Cette question demande ton avis. Tu peux exprimer une opinion équilibrée. "
        )
        prefix = "Question d'opinion: "
        suffix = "\nMon point de vue: "

    elif question_type == "how-to":
        # Pour les questions de procédure
        temperature = 0.7
        context = (
            "Cette question demande une procédure. Explique les étapes clairement. "
        )
        prefix = "Comment: "
        suffix = "\nVoici les étapes: "

    else:
        # Par défaut
        context = "Réponds à cette question de manière claire et concise. "
        prefix = "Question: "
        suffix = "\nRéponse: "

    # Construire le prompt final
    full_prompt = f"{context}{prefix}{prompt.strip()}{suffix}"

    # Tokenize et convertir en tensor
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    # Configurer les paramètres de génération spécifiques au type de question
    gen_kwargs = {
        "max_length": max_length + len(inputs["input_ids"][0]),
        "temperature": temperature,
        "top_p": 0.85 if question_type == "factual" else top_p,
        "repetition_penalty": 1.5 if question_type == "factual" else repetition_penalty,
        "do_sample": not question_type
        == "factual",  # Pour les questions factuelles, désactiver le sampling
        "top_k": 10 if question_type == "factual" else 30,
        "no_repeat_ngram_size": 3,
        "num_beams": 5 if question_type == "factual" else 3,
        "early_stopping": True,
        "length_penalty": 0.5 if question_type == "factual" else 1.0,
    }

    # Générer la réponse
    set_seed(random.randint(1, 1000))  # Pour la reproductibilité mais avec variation
    output_sequences = model.generate(**inputs, **gen_kwargs)

    # Décoder et nettoyer la réponse
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    try:
        # Extraire uniquement la partie réponse en utilisant le séparateur
        response = generated_text.split(suffix)[-1].strip()
    except:
        # En cas d'échec du split, utiliser tout le texte généré
        response = generated_text.strip()

    # Nettoyer la réponse
    response = clean_response(response)

    # Post-traitement spécifique pour les questions factuelles
    if question_type == "factual":
        # Pour les questions factuelles, prendre seulement la première phrase
        response = extract_first_sentence(response)

    return response, question_type


def extract_first_sentence(text):
    """Extrait la première phrase complète d'un texte"""
    # Chercher la fin de la première phrase
    sentence_end = re.search(r"[.!?](\s|$)", text)
    if sentence_end:
        end_pos = sentence_end.end()
        return text[:end_pos].strip()

    # Si pas de ponctuation de fin de phrase trouvée, retourner le texte entier
    return text.strip()


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

    # Ajouté: vérifier si la réponse est une question qui répète la question originale
    if re.match(r"^(qu|qui|que|quoi|comment|pourquoi|quand|où)", response.lower()):
        if (
            "?" in response[:50]
        ):  # Si c'est une question dans les 50 premiers caractères
            return "Désolé, je ne dispose pas de suffisamment d'informations précises pour répondre à cette question."

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
            # Faire jusqu'à 3 tentatives pour obtenir une réponse satisfaisante
            max_attempts = 3
            response = None
            question_type = None
            
            # Pour la première tentative, définir une longueur standard
            initial_max_length = 150
            
            for attempt in range(max_attempts):
                # Générer la réponse avec l'aide du réseau d'analyse
                # Utiliser la longueur appropriée selon le type de question (uniquement si déjà déterminé)
                current_max_length = 80 if question_type == "factual" else initial_max_length
                
                response, question_type = generate_response(
                    user_input,
                    model,
                    tokenizer,
                    analyzer,
                    max_length=current_max_length,
                    temperature=0.7 - (attempt * 0.2),  # Réduction de température à chaque tentative
                    repetition_penalty=1.2 + (attempt * 0.3)  # Augmentation de pénalité à chaque tentative
                )

                # Vérifier si la réponse est acceptable
                if is_response_acceptable(response, question_type):
                    break  # Réponse acceptable

                # Si ce n'est pas la dernière tentative, informer l'utilisateur
                if attempt < max_attempts - 1:
                    print(f"(Amélioration de la réponse, tentative {attempt+2}/{max_attempts}...)")

            # Si après toutes les tentatives, la réponse est toujours mauvaise pour une question factuelle
            if question_type == "factual" and not is_response_acceptable(response, question_type):
                response = "Je ne dispose pas d'informations précises sur ce sujet. Pour les questions factuelles, je peux répondre avec certitude sur des sujets comme les capitales des pays, populations approximatives, ou faits historiques importants."

            elapsed_time = time.time() - start_time

            print(f"\n🤖 Réponse (type: {question_type}, temps: {elapsed_time:.2f}s):")
            print(response)
            print("\n" + "=" * 50)

        except Exception as e:
            print(f"❌ Erreur lors de la génération de la réponse: {str(e)}")
            import traceback

            traceback.print_exc()

    print("\nAu revoir! 👋")

    # Sauvegarder les améliorations du réseau d'analyse (en production)
    # torch.save(analyzer.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "question_analyzer.pt"))


def is_response_acceptable(response, question_type):
    """Vérifie si une réponse est acceptable selon son type"""
    # Pour les questions factuelles, critères plus stricts
    if question_type == "factual":
        # La réponse ne doit pas être une question
        if "?" in response[:50]:
            return False
        # Doit contenir des verbes d'affirmation typiques
        if not re.search(
            r"\b(est|sont|a|ont|était|étaient|fait|se trouve|se situe|compte|contient)\b",
            response,
            re.IGNORECASE,
        ):
            return False
        # Longueur minimale et maximale
        if len(response) < 10 or len(response) > 150:
            return False

    # Critères généraux pour tous les types de questions
    # Pas de mots-clés problématiques
    if any(word in response for word in ["Naissances", "Portail", "Catégorie"]):
        return False
    # Pas de réponse qui finit ou commence par une question
    if response.endswith("?") or response.startswith("?"):
        return False
    # Longueur minimale
    if len(response) < 10:
        return False

    return True


if __name__ == "__main__":
    main()
