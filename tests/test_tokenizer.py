
import os
from transformers import AutoTokenizer
import torch

def test_tokenizer(model_path="trained_llm"):
    """
    Teste le tokenizer pour vérifier son bon fonctionnement et sa cohérence.
    """
    print("\n" + "="*50)
    print("🔍 TEST DU TOKENIZER")
    print("="*50)
    
    # 1. Vérifier que le tokenizer peut être chargé
    print("\n1️⃣ Chargement du tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✅ Tokenizer chargé avec succès")
        print(f"   Type de tokenizer: {type(tokenizer).__name__}")
        print(f"   Taille du vocabulaire: {len(tokenizer)}")
        print(f"   Tokens spéciaux: {tokenizer.all_special_tokens}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du tokenizer: {str(e)}")
        return
    
    # 2. Tester les tokens spéciaux
    print("\n2️⃣ Vérification des tokens spéciaux...")
    required_special_tokens = ['<|bos|>', '<|eos|>', '<|pad|>']
    missing_tokens = [token for token in required_special_tokens if token not in tokenizer.all_special_tokens]
    
    if missing_tokens:
        print(f"⚠️ Tokens spéciaux manquants: {missing_tokens}")
        
        # Ajouter les tokens manquants et sauvegarder
        print("   Tentative d'ajout des tokens manquants...")
        special_tokens_dict = {}
        
        if '<|bos|>' not in tokenizer.all_special_tokens:
            special_tokens_dict['bos_token'] = '<|bos|>'
        if '<|eos|>' not in tokenizer.all_special_tokens:
            special_tokens_dict['eos_token'] = '<|eos|>'
        if '<|pad|>' not in tokenizer.all_special_tokens:
            special_tokens_dict['pad_token'] = '<|pad|>'
            
        tokenizer.add_special_tokens(special_tokens_dict)
        print(f"✅ Tokens spéciaux ajoutés: {tokenizer.all_special_tokens}")
        
        # Sauvegarder le tokenizer modifié
        tokenizer.save_pretrained(model_path)
        print(f"✅ Tokenizer sauvegardé dans {model_path}")
    else:
        print("✅ Tous les tokens spéciaux sont présents")
    
    # 3. Tester le tokenizer sur des exemples français
    print("\n3️⃣ Test de tokenisation sur des exemples français...")
    test_sentences = [
        "La capitale de la France est Paris.",
        "L'intelligence artificielle révolutionne le monde.",
        "Comment fonctionnent les réseaux de neurones?",
        "Je suis un modèle de langage entraîné en français."
    ]
    
    for sentence in test_sentences:
        tokens = tokenizer.tokenize(sentence)
        token_ids = tokenizer.encode(sentence)
        decoded = tokenizer.decode(token_ids)
        
        print(f"\nPhrase: {sentence}")
        print(f"Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        print(f"IDs: {token_ids[:10]}{'...' if len(token_ids) > 10 else ''}")
        print(f"Reconstruction: {decoded}")
        
        if sentence.strip() not in decoded:
            print("⚠️ La reconstruction ne contient pas la phrase originale!")
        else:
            print("✅ Reconstruction correcte")
    
    # 4. Tester la cohérence
    print("\n4️⃣ Test de cohérence...")
    test_phrase = "La capitale de la France est Paris."
    
    # Tokeniser et décoder plusieurs fois pour vérifier la cohérence
    results = []
    for _ in range(5):
        tokens = tokenizer.encode(test_phrase)
        decoded = tokenizer.decode(tokens)
        results.append((tokens, decoded))
    
    # Vérifier si tous les résultats sont identiques
    all_tokens_identical = all(result[0] == results[0][0] for result in results)
    all_decoded_identical = all(result[1] == results[0][1] for result in results)
    
    if all_tokens_identical and all_decoded_identical:
        print("✅ Le tokenizer est cohérent (donne toujours le même résultat)")
    else:
        print("⚠️ Le tokenizer n'est pas cohérent!")
        if not all_tokens_identical:
            print("   Les tokens diffèrent entre les exécutions")
        if not all_decoded_identical:
            print("   Les décodages diffèrent entre les exécutions")
    
    # 5. Tester avec les tokens spéciaux
    print("\n5️⃣ Test avec tokens spéciaux...")
    
    special_test = f"{tokenizer.bos_token}Question: Quelle est la capitale de la France? Réponse:{tokenizer.eos_token}"
    special_tokens = tokenizer.tokenize(special_test)
    special_ids = tokenizer.encode(special_test)
    special_decoded = tokenizer.decode(special_ids)
    
    print(f"Phrase avec tokens spéciaux: {special_test}")
    print(f"Tokens: {special_tokens[:15]}{'...' if len(special_tokens) > 15 else ''}")
    print(f"Reconstruction: {special_decoded}")

    if tokenizer.bos_token in special_decoded and tokenizer.eos_token in special_decoded:
        print("✅ Les tokens spéciaux sont correctement reconnus et reconstruits")
    else:
        print("⚠️ Problème avec les tokens spéciaux dans la reconstruction!")
    
    print("\n" + "="*50)
    print("🏁 FIN DU TEST")
    print("="*50)

if __name__ == "__main__":
    # Chemin par défaut pour le modèle
    default_path = "trained_llm"
    
    # Vérifier si le dossier existe
    if not os.path.exists(default_path):
        print(f"⚠️ Le dossier {default_path} n'existe pas, recherche d'alternatives...")
        
        # Rechercher d'autres emplacements possibles
        alternatives = [
            "models/trained",
            "../trained_llm",
            "output/llm"
        ]
        
        for alt_path in alternatives:
            if os.path.exists(alt_path):
                print(f"✅ Alternative trouvée: {alt_path}")
                default_path = alt_path
                break
        else:
            # Si aucune alternative n'est trouvée, utiliser un modèle préentraîné
            print("⚠️ Aucun dossier de modèle trouvé, utilisation d'un modèle préentraîné")
            default_path = "dbddv01/gpt2-french-small"
    
    # Lancer le test
    test_tokenizer(default_path)
