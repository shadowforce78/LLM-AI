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

        # Initialisation explicite avec des biais appropriés
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialise les poids du réseau avec des biais pour les types de questions courants"""
        # Assurer une meilleure détection des questions factuelles vs opinions
        torch.nn.init.xavier_uniform_(self.fc.weight)
        # Bias factuel plus élevé pour reconnaître les questions factuelles
        self.fc.bias.data = torch.tensor([0.5, -0.3, -0.2, -0.2, -0.2, -0.1], dtype=torch.float)

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
    factual_keywords = ["quelle", "qui", "quand", "où", "combien", "pourquoi", "capitale", 
                      "population", "superficie", "président", "nombre", "habitant", "est"]
    opinion_keywords = ["penses", "crois", "avis", "opinion", "préfères", "meilleur", "pire"]
    howto_keywords = ["comment", "procédure", "étapes", "faire", "créer", "développer", "fabriquer"]
    
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
            analyzer.embedding.weight.data[idx] = torch.tensor([0.5 if j == 20 else 0.1 * random.random() 
                                                         for j in range(analyzer.embedding.weight.data[idx].shape[0])],
                                                        dtype=torch.float)
        elif idx < 200:  # Opinion words
            # Biais vers la catégorie opinion (indice 1)
            analyzer.embedding.weight.data[idx] = torch.tensor([0.5 if j == 10 else 0.1 * random.random() 
                                                         for j in range(analyzer.embedding.weight.data[idx].shape[0])],
                                                        dtype=torch.float)
        else:  # How-to words
            # Biais vers la catégorie how-to (indice 2)
            analyzer.embedding.weight.data[idx] = torch.tensor([0.5 if j == 30 else 0.1 * random.random() 
                                                         for j in range(analyzer.embedding.weight.data[idx].shape[0])],
                                                        dtype=torch.float)


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
        temperature *= 0.5  # Réduction plus importante de la température
        max_length = min(max_length, 80)  # Réponses plus courtes et précises
        prefix = "Question factuelle: "
        suffix = "\nRéponse précise et factuelle: "
        # Pour les questions factuelles courantes, ajouter un contexte qui guide vers la bonne réponse
        context_additions = {
            "capitale france": "La capitale de la France est ",
            "superficie france": "La superficie de la France est d'environ ",
            "population france": "La France compte environ ",
            "habitants france": "La France compte environ ",
            "président france": "Le président de la France est "
        }
        
        # Vérifier si la question contient certains mots-clés pour ajouter un contexte spécifique
        prompt_lower = prompt.lower()
        context_addition = ""
        for key, value in context_additions.items():
            if all(word in prompt_lower for word in key.split()):
                context_addition = value
                break
    else:
        # Paramètres pour les autres types de questions
        prefix = "Question: "
        suffix = "\nRéponse: "
        context_addition = ""

    # Construire le prompt complet avec instructions pour le modèle
    context = "Tu es un assistant IA français qui répond de manière claire et précise. "
    
    # Ajouter des instructions spécifiques selon le type de question
    if question_type == 'factual':
        context += "Cette question demande des faits précis. " + context_addition 
        # Ajouter un exemple explicite pour les questions factuelles
        if "capitale" in prompt.lower():
            context += "Par exemple: La capitale de la France est Paris. "
        elif "habitants" in prompt.lower() or "population" in prompt.lower():
            context += "Par exemple: La population de la France est d'environ 67 millions d'habitants. "
    
    full_prompt = f"{context}{prefix}{prompt.strip()}{suffix}"

    # Tokenize et convertir en tensor
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    # Configurer les paramètres de génération
    gen_kwargs = {
        "max_length": max_length + len(inputs["input_ids"][0]),
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty * 1.3,  # Augmentation plus forte de la pénalité de répétition
        "do_sample": True,
        "top_k": 20 if question_type == 'factual' else 30,  # Plus restrictif pour les questions factuelles
        "no_repeat_ngram_size": 3,
        "num_beams": 5 if question_type == 'factual' else 3,
        "early_stopping": True,
        "length_penalty": 0.7 if question_type == 'factual' else 1.0,  # Préférer des réponses plus courtes pour les faits
    }

    # Générer la réponse
    set_seed(random.randint(1, 1000))
    output_sequences = model.generate(**inputs, **gen_kwargs)

    # Décoder et nettoyer la réponse
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    response = generated_text.split(suffix)[-1].strip()
    
    # Nettoyer la réponse
    response = clean_response(response)
    
    # Pour les questions factuelles, appliquer un filtrage supplémentaire
    if question_type == 'factual':
        response = filter_factual_response(response, prompt)
    
    return response, question_type

def filter_factual_response(response, question):
    """Filtre spécial pour les réponses factuelles"""
    # Simplifier pour n'obtenir que la première phrase substantielle
    sentences = re.split(r'([.!?]\s+)', response)
    if len(sentences) > 2:  # Au moins une phrase complète (texte + ponctuation)
        first_sentence = sentences[0] + (sentences[1] if len(sentences) > 1 else '.')
        if len(first_sentence) > 15:  # Si la première phrase est substantielle
            return first_sentence.strip()
    
    # Vérifier si la réponse contient des signes de réponse factuelle
    if re.search(r'(est|sont|était|fait|situe|compte|mesure|contient|existait)', response):
        return response
        
    # Sinon, message de secours
    return "Je n'ai pas suffisamment d'informations précises pour répondre à cette question factuelle."


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
            # Faire jusqu'à 3 tentatives pour obtenir une réponse satisfaisante
            max_attempts = 3
            for attempt in range(max_attempts):
                # Générer la réponse avec l'aide du réseau d'analyse
                response, question_type = generate_response(
                    user_input, model, tokenizer, analyzer, 
                    max_length=150, 
                    temperature=0.7 - (attempt * 0.1),  # Réduire la température à chaque tentative
                    repetition_penalty=1.2 + (attempt * 0.2)  # Augmenter la pénalité de répétition
                )
                
                # Vérifier si la réponse est acceptable
                if (len(response) >= 10 and 
                    not any(word in response for word in ["Naissances", "Portail", "Catégorie"]) and
                    not response.endswith("?") and
                    not response.startswith("?")):
                    break  # Réponse acceptable
                    
                # Si ce n'est pas la dernière tentative, informer l'utilisateur
                if attempt < max_attempts - 1:
                    print(f"(Amélioration de la réponse, tentative {attempt+2}/{max_attempts}...)")
            
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


if __name__ == "__main__":
    main()
