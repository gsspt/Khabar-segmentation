#!/usr/bin/env bash
# session-end.sh — Sauvegarde et pousse le travail en cours vers GitHub.
# Usage : ./scripts/session-end.sh [message optionnel]
# Utiliser à la fin de chaque session de travail (VSCode, terminal, smartphone).

set -euo pipefail

BRANCH=$(git branch --show-current)
MSG="${1:-"auto: sauvegarde $(date '+%Y-%m-%d %H:%M') [$(hostname -s 2>/dev/null || echo env)]"}"

# Vérifier qu'on est bien dans le bon dépôt
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "❌ Pas dans un dépôt git." >&2
    exit 1
fi

# Configurer le remote avec le token si disponible
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    REMOTE_URL=$(git remote get-url origin | sed 's|https://[^@]*@|https://|')
    git remote set-url origin "https://${GITHUB_TOKEN}@${REMOTE_URL#https://}"
fi

# Vérifier s'il y a des changements à committer
if git diff --quiet && git diff --cached --quiet && [[ -z $(git ls-files --others --exclude-standard) ]]; then
    echo "✓ Rien à committer."
else
    git add -A
    git commit -m "$MSG"
    echo "✓ Commit : $MSG"
fi

# Pousser les commits non encore envoyés
UNPUSHED=$(git rev-list "origin/${BRANCH}..HEAD" --count 2>/dev/null || echo "?")
if [[ "$UNPUSHED" == "0" ]]; then
    echo "✓ Déjà à jour sur origin/$BRANCH."
else
    git push -u origin "$BRANCH"
    echo "✓ Push vers origin/$BRANCH ($UNPUSHED commit(s))."
fi
