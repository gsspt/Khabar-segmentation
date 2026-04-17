#!/usr/bin/env bash
# session-start.sh — Synchronise le dépôt local avec GitHub au début d'une session.
# Usage : ./scripts/session-start.sh
# Utiliser au démarrage de chaque session de travail.

set -euo pipefail

BRANCH=$(git branch --show-current)

if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "❌ Pas dans un dépôt git." >&2
    exit 1
fi

# Injecter le token uniquement si le remote pointe directement vers GitHub
# (pas dans Claude Code web où un proxy local gère l'auth)
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    REMOTE_URL=$(git remote get-url origin)
    if [[ "$REMOTE_URL" == https://github.com/* ]]; then
        git remote set-url origin "https://${GITHUB_TOKEN}@github.com/${REMOTE_URL#https://github.com/}"
    fi
fi

echo "↓ Synchronisation depuis origin/$BRANCH..."

# Stash les changements locaux non committés pour éviter les conflits
STASHED=false
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "  → Changements locaux détectés, mise en attente (stash)..."
    git stash push -m "session-start auto-stash $(date '+%Y-%m-%d %H:%M')"
    STASHED=true
fi

# Pull avec rebase pour un historique propre
git pull --rebase origin "$BRANCH"

# Restaurer les changements mis en attente
if [[ "$STASHED" == true ]]; then
    echo "  → Restauration des changements locaux..."
    git stash pop
fi

echo "✓ Dépôt à jour sur origin/$BRANCH."
