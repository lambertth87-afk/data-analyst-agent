# data-analyst — agent analyste de données conversationnel

Agent construit avec le [SDK Agents de Mistral](https://docs.mistral.ai/studio/agents/overview),
déployé sur Mistral Cloud en déploiement géré (Mistral-managed).

Vous lui donnez un lien vers un fichier CSV de votre Google Drive et lui posez des
questions en langage naturel. Il récupère le fichier avec **votre** connexion
Google Drive (connecteur `google_drive_mcp`, exécuté en votre nom), puis répond
avec des chiffres réels calculés par des **outils fixes** — aucune exécution de
code arbitraire.

## Outils (tous en lecture seule)

| Outil | Rôle |
| --- | --- |
| `load_csv` | Charge un CSV (base64 ou texte) renvoyé par le connecteur Drive ; renvoie un `dataset_id`, le schéma des colonnes et un aperçu |
| `describe_columns` | Décrit chaque colonne : type, valeurs manquantes, unicités, min/max/moyenne/médiane/somme ou valeurs les plus fréquentes |
| `filter_rows` | Filtre les lignes (`eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `contains`, `in`, `is_null`, `not_null`) et renvoie le nombre de correspondances + un échantillon |
| `aggregate_by_group` | Agrégats par groupe (`count`, `nunique`, `sum`, `mean`, `median`, `min`, `max`) avec filtres optionnels et tri |

Le connecteur Google Drive fournit `search_files`, `get_file_metadata` et
`download_file_content` (export CSV pour un Google Sheet). L'agent n'a **pas de
sandbox** : il ne peut ni exécuter du shell, ni écrire des fichiers, ni modifier
vos données Drive.

## Mémoire

La conversation entière vit dans une **session** stockée par Mistral : chaque
question de suivi réutilise le `dataset_id` chargé et les filtres déjà discutés,
sans que vous ayez à vous réexpliquer. Si la session est recyclée, l'agent
retélécharge le fichier depuis Drive et le recharge automatiquement.

## Développement local

```bash
uv sync --all-groups
uv run pytest
uv run agents info        # vérifie l'agent : outils, connecteur, sandbox
uv run --env-file .env agents chat   # chat local (nécessite MISTRAL_API_KEY)
```

## Déploiement géré

Le déploiement est **hardened** (obligatoire : l'agent agit au nom de
l'utilisateur de la session via le connecteur Drive). Deux voies :

1. CLI documenté : `uv run agents deploy` (nécessite `MISTRAL_API_KEY` avec
   accès Workflows, un dépôt GitHub et le connecteur GitHub connecté).
2. Via l'API Studio : `deploy_workflow_deployment` avec `hardened=true`,
   qui build le `Dockerfile` du commit poussé (CMD `agents run-worker`).

Une fois déployé, l'agent est disponible dans Vibe :
`https://chat.mistral.ai/managed-agents/data-analyst?source=agents_api`.

## Structure

- `main.py` — l'agent (modèle, instructions, outils, connecteur)
- `tools.py` — les quatre outils d'analyse fixes (pandas, déterministes)
- `tests/` — tests des outils sans clé API
- `Dockerfile` — image du worker (générée selon le modèle du SDK, éditable)
