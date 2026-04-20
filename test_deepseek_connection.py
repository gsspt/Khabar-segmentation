#!/usr/bin/env python3
"""Test la connexion à l'API Deepseek"""

import os
import requests
from dotenv import load_dotenv

# Charger .env
load_dotenv()
api_key = os.getenv('DEEPSEEK_API_KEY')

print(f"[1] Vérifier la clé API")
if api_key:
    print(f"  ✓ Clé trouvée: {api_key[:20]}...")
else:
    print(f"  ✗ Clé non trouvée dans .env")
    exit(1)

print(f"\n[2] Tester la connexion de base")
try:
    response = requests.get('https://api.deepseek.com', timeout=5)
    print(f"  ✓ Connexion HTTP OK (status: {response.status_code})")
except Exception as e:
    print(f"  ✗ Erreur: {e}")
    print(f"  → Vérifier: proxy, VPN, firewall, certificat SSL")

print(f"\n[3] Tester l'authentification API")
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Say 'Hello' in Arabic"}
    ],
    "temperature": 0.3,
    "max_tokens": 100,
}

try:
    response = requests.post(
        "https://api.deepseek.com/chat/completions",
        json=payload,
        headers=headers,
        timeout=10
    )

    print(f"  Status: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        if 'choices' in result:
            print(f"  ✓ API fonctionne!")
            print(f"  Réponse: {result['choices'][0]['message']['content']}")
        else:
            print(f"  ✗ Réponse invalide: {result}")
    else:
        print(f"  ✗ Erreur {response.status_code}")
        print(f"  Détails: {response.text[:200]}")

except requests.exceptions.Timeout:
    print(f"  ✗ Timeout - l'API met trop de temps à répondre")
except requests.exceptions.ConnectionError as e:
    print(f"  ✗ Erreur de connexion: {e}")
    print(f"  → Vérifier proxy/VPN/firewall")
except Exception as e:
    print(f"  ✗ Erreur: {e}")

print(f"\n[4] Suggestions")
print(f"  Si erreur de connexion:")
print(f"    - Vérifier la clé API Deepseek")
print(f"    - Vérifier la connexion internet")
print(f"    - Vérifier proxy/VPN/firewall")
print(f"    - Essayer depuis un autre réseau")
print(f"\n  Documentation: https://platform.deepseek.com/")
