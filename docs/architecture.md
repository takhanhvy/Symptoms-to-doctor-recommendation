# Architecture

## Vue d'ensemble
L'application propose une orientation medicale a partir d'une description libre des symptomes. Le flux principal combine un referentiel medical, un encodage semantique des textes et une generation de synthese par IA.

## Composants
- UI (Streamlit): formulaire utilisateur, affichage des recommandations et de la synthese.
- Donnees: referentiel medical nettoye et consolide (data/processed).
- NLP local: embeddings de phrases via SBERT pour comparer la description utilisateur au referentiel.
- GenAI: generation d'une synthese explicative basee sur les resultats.

## Flux de traitement
1. L'utilisateur saisit ses symptomes et informations associees.
2. L'application construit un texte utilisateur structure.
3. Le texte est vectorise (embeddings SBERT).
4. Le referentiel est vectorise et compare a l'entree utilisateur via similarite cosinus.
5. Les meilleures correspondances sont retournees comme recommandations.
6. Une synthese explicative est generee par IA a partir du contexte et des resultats.

## Schema de flux (ASCII)
```
[Utilisateur]
     |
     v
[UI Streamlit] --> [Construction texte]
                        |
                        v
                  [Embeddings SBERT]
                        |
                        v
            [Similarite cosinus + Top K]
                        |
                        v
              [Recommandations]
                        |
                        v
                 [Synthese GenAI]
                        |
                        v
                   [Affichage]
```

## Caching et performances
- Mise en cache des embeddings du referentiel pour eviter des recalculs couteux.
- Cache de reponses GenAI afin de limiter les appels et accelerer l'affichage.

## Limites
- Le systeme ne fournit pas de diagnostic medical.
- Les recommandations sont indicatives et doivent etre validees par un professionnel.
