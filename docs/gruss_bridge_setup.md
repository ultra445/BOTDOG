# Gruss Excel Bridge Setup

Par defaut, cette passerelle lit uniquement les donnees Excel ecrites par Gruss Betting Assistant. Les scripts de lecture et de dry-run ne placent aucun ordre et ne modifient aucune cellule de trigger Gruss.

## Preparation

1. Ouvrir Gruss Betting Assistant.
2. Ouvrir Microsoft Excel.
3. Ouvrir le classeur:
   `C:\betfair-dogbot\gruss_bridge\dogbot_gruss.xlsx`
4. Dans Gruss, lier le marche WIN a l'onglet `WIN`.
5. Dans Gruss, lier le marche PLACE a l'onglet `PLACE`.

Gruss doit ecrire les prix a partir de `A1` dans chaque onglet.

## Dump diagnostic

Depuis `C:\betfair-dogbot`, lancer:

```powershell
python scripts/dump_gruss_sheet.py
```

Le script tente d'abord de se connecter au workbook deja ouvert. S'il ne le trouve pas, il ouvre:

`C:\betfair-dogbot\gruss_bridge\dogbot_gruss.xlsx`

Il lit ensuite les 80 premieres lignes et les 50 premieres colonnes des onglets `WIN` et `PLACE`.

## Fichiers generes

Les CSV diagnostics sont ecrits ici:

- `data/gruss_dumps/win_dump.csv`
- `data/gruss_dumps/place_dump.csv`

Ces dumps servent uniquement a observer la structure reelle des feuilles Gruss avant de definir un mapping stable.

## Snapshot live

Pour lire le workbook Excel Gruss deja ouvert et afficher un diagnostic parse WIN/PLACE:

```powershell
python scripts\read_gruss_snapshot.py
```

Ce script lit uniquement Excel. Il affiche les resumes des marches, la validation WIN/PLACE et les runners appaires par trap.

## Surveillance dry-run

Pour surveiller le flux Excel Gruss ouvert et ecrire des snapshots diagnostics sans envoyer d'ordre:

```powershell
python scripts\watch_gruss_feed.py --interval 1 --max-ticks 120
```

Les lignes valides sont ajoutees dans:

`data/gruss_live_snapshots.csv`

## Moteur dry-run Gruss

Le mode Gruss doit etre active explicitement par configuration:

```powershell
$env:DOGBOT_DATA_PROVIDER="gruss_excel"
$env:DOGBOT_ORDER_PROVIDER="gruss_excel_dryrun"
```

Evaluation unique:

```powershell
python scripts\run_gruss_dryrun_once.py
```

Surveillance avec evaluation unique a l'approche de T-2s:

```powershell
python scripts\watch_gruss_dryrun.py --interval 1 --max-ticks 120
```

Ces scripts utilisent les strategies existantes uniquement en dry-run. Ils n'envoient aucun ordre et n'ecrivent pas dans Excel.

Les lignes `DRYRUN` generees depuis Gruss dans `data/trades_YYYYMMDD.csv` sont enrichies avec les diagnostics Gruss: provider, market ids WIN/PLACE, parent id, countdown, tradable, meilleures cotes WIN/PLACE, place theorique, EV place et titres/path Gruss quand ces valeurs sont disponibles.

## Preview du provider d'ordres Gruss reel

Le provider reel est desactive par defaut. La commande de preview exige une
activation explicite, mais force toujours le mode preview et n'ecrit aucune
cellule Excel:

```powershell
$env:DOGBOT_DATA_PROVIDER="gruss_excel"
$env:DOGBOT_ORDER_PROVIDER="gruss_excel_real"
$env:DOGBOT_GRUSS_ENABLE_REAL_ORDERS="true"
$env:DOGBOT_GRUSS_REAL_PREVIEW="true"
python scripts\run_gruss_real_preview_once.py --market-type PLACE --trap 1 --side BACK --stake 2
```

Le plan exact des cellules qui seraient ecrites est affiche et l'essai est
journalise dans:

`data/gruss_real_order_attempts.csv`

Le layout standard configure est `Q=Trigger`, `R=Odds`, `S=Stake`, avec le
trigger ecrit en dernier. Les ordres `SP_MOC` sont traduits en `BACKSP` ou
`LAYSP` et exigent aussi un prix limite explicite. Une ecriture reelle exige en
plus:

```powershell
$env:DOGBOT_GRUSS_TRIGGER_LAYOUT_CONFIRMED="true"
$env:DOGBOT_GRUSS_REAL_PREVIEW="false"
```

Ne pas activer ces deux variables avant d'avoir confirme manuellement le layout
du workbook Gruss. Aucun script dry-run existant n'active le provider reel.

Surveillance continue du provider reel en preview-only:

```powershell
$env:DOGBOT_DATA_PROVIDER="gruss_excel"
$env:DOGBOT_ORDER_PROVIDER="gruss_excel_real"
$env:DOGBOT_GRUSS_ENABLE_REAL_ORDERS="false"
$env:DOGBOT_GRUSS_REAL_PREVIEW="true"
python scripts\watch_gruss_real_preview.py --interval 1 --max-ticks 120
```

Ce watcher refuse de demarrer si les ordres reels sont armes. Il attend une
course valide et tradable, puis evalue une seule fois lorsque le countdown est
inferieur ou egal a 2 secondes. Il journalise uniquement les previews dans
`data/gruss_real_order_attempts.csv`.

## Ecriture de preparation sans trigger

Le mode `write-no-trigger` ecrit uniquement les cellules de preparation Gruss
connues (`R=Odds`, `S=Stake`). Il ne touche jamais la cellule `Q=Trigger`, donc
les valeurs de trigger `BACK`, `LAY`, `BACKSP` et `LAYSP` ne sont jamais
ecrites. Le side et la strategie restent visibles dans le journal diagnostic.
Le champ `intended_trigger` montre la valeur qui aurait ete utilisee en mode
reel, sans jamais l'ecrire dans Excel.

```powershell
$env:DOGBOT_DATA_PROVIDER="gruss_excel"
$env:DOGBOT_ORDER_PROVIDER="gruss_excel_real"
$env:DOGBOT_GRUSS_WRITE_NO_TRIGGER="true"
$env:DOGBOT_GRUSS_ENABLE_REAL_ORDERS="false"
$env:DOGBOT_GRUSS_REAL_PREVIEW="false"
python scripts\watch_gruss_write_no_trigger.py --interval 1 --max-ticks 120
```

Ce mode n'exige pas l'armement reel. Meme si
`DOGBOT_GRUSS_ENABLE_REAL_ORDERS=true` est present, le provider protege
toujours la cellule Trigger. Il refuse aussi l'ecriture si cette cellule
contient deja une valeur. Les tentatives sont journalisees avec:

En mode write-no-trigger, `DOGBOT_GRUSS_REAL_PREVIEW` absent, vide ou egal a
`false` est accepte. Seule la valeur `true` est refusee.

- `status=GRUSS_WRITE_NO_TRIGGER`
- `reason=no_trigger_written`
- `cells_written=R...;S...`
- `trigger_written=False`

Les courses traitees sont conservees dans
`data/gruss_write_no_trigger_processed.csv`.

Tous les signaux produits au meme tick sont traites comme un batch. La course
n'est marquee comme traitee qu'apres toutes les tentatives du batch; le tick
suivant est ensuite ignore.

## Limites runtime du vrai mode reel

Ces limites s'appliquent uniquement au vrai mode reel avec trigger. Elles
n'affectent ni dry-run, ni preview, ni write-no-trigger:

```powershell
$env:DOGBOT_GRUSS_REAL_TEST_MODE="true"
$env:DOGBOT_GRUSS_REAL_MAX_ORDERS="1"
$env:DOGBOT_GRUSS_REAL_MAX_STAKE="1"
```

En test mode, les valeurs par defaut sont un ordre reel par course et une mise
maximale de 1. Une mise superieure est refusee, jamais plafonnee silencieusement,
avec `stake_above_real_test_limit`. Les ordres au-dela de la limite par course
sont refuses avec `max_orders_reached`.

Hors test mode, les limites restent applicables si
`DOGBOT_GRUSS_REAL_MAX_ORDERS` ou `DOGBOT_GRUSS_REAL_MAX_STAKE` sont definies.
Si elles sont absentes ou vides, aucune limite runtime supplementaire n'est
appliquee.

## Premier test reel limite

Le watcher dedie au premier test reel exige un armement complet et refuse de
demarrer si une limite est absente ou trop large:

```powershell
$env:DOGBOT_DATA_PROVIDER="gruss_excel"
$env:DOGBOT_ORDER_PROVIDER="gruss_excel_real"
$env:DOGBOT_GRUSS_ENABLE_REAL_ORDERS="true"
$env:DOGBOT_GRUSS_REAL_TEST_MODE="true"
$env:DOGBOT_GRUSS_REAL_MAX_ORDERS="1"
$env:DOGBOT_GRUSS_REAL_MAX_STAKE="1"
$env:DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="1"
$env:DOGBOT_GRUSS_REAL_PREVIEW="false"
$env:DOGBOT_GRUSS_WRITE_NO_TRIGGER="false"
$env:DOGBOT_GRUSS_TRIGGER_LAYOUT_CONFIRMED="true"
python scripts\watch_gruss_real_test.py --interval 1
```

Ce script peut ecrire un vrai trigger Excel. Il attend une course valide,
tradable et connue jusqu'a T-2s, puis autorise au maximum un ordre reel par
course. Toute mise superieure a la limite est refusee sans plafonnement. Toutes
les tentatives restent journalisees dans `data/gruss_real_order_attempts.csv`.

`DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE` est optionnelle et exclusivement lue par
`watch_gruss_real_test.py`. Lorsqu'elle vaut `1`, la mise du signal est
remplacee par `1` juste avant l'unique tentative reelle. Elle ne s'applique
jamais au dry-run, preview ou write-no-trigger. Si elle depasse
`DOGBOT_GRUSS_REAL_MAX_STAKE`, l'ordre est refuse avec
`stake_above_real_test_limit`. Le journal conserve `stake_original`,
`stake_used` et `stake_forced`.

### Test force du canal BSP PLACE

Le watcher reel test peut ignorer les strategies normales et construire un
unique ordre de verification `BACK PLACE SP_MOC`. Il selectionne le runner
ayant le plus faible `best_back` PLACE disponible et utilise le mapping Gruss
existant `BACKSP`:

```powershell
$env:DOGBOT_GRUSS_FORCE_TEST_BSP_PLACE="true"
$env:DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="1"
python scripts\watch_gruss_real_test.py --interval 1
```

Ce mode exige aussi tout l'armement du premier test reel, avec
`DOGBOT_GRUSS_REAL_MAX_ORDERS=1`, `DOGBOT_GRUSS_REAL_MAX_STAKE=1` et le layout
trigger confirme. Il refuse de demarrer hors real-test mode. Le journal ajoute
`force_test_bsp_place`, `selected_reason`, `selected_runner`, `selected_trap`
et `selected_place_odds`.

### Test force BACK PLACE LIMIT

Pour tester un BACK PLACE classique au meilleur prix LAY disponible:

```powershell
$env:DOGBOT_GRUSS_FORCE_TEST_BSP_PLACE="false"
$env:DOGBOT_GRUSS_FORCE_TEST_BACK_PLACE_LIMIT="true"
$env:DOGBOT_GRUSS_REAL_TEST_FORCE_STAKE="1"
python scripts\watch_gruss_real_test.py --interval 1
```

Ce mode est mutuellement exclusif avec le test BSP. Il selectionne le runner
PLACE ayant le plus faible `best_back`, exige un `best_lay` valide, puis ecrit
un unique ordre LIMIT avec `R{row}=best_lay`, `S{row}=1` et `Q{row}=BACK`.
Les strategies normales ne sont pas evaluees. Le journal conserve
`selected_place_back_odds`, `selected_place_lay_odds`, `price_used`, le trigger
ecrit et le resultat du nettoyage differe de Q.

Pour inspecter les cellules trigger PLACE sans aucune ecriture:

```powershell
python scripts\inspect_gruss_trigger_cells.py
```

Le layout configure actuellement `R{row}` pour Odds/SP, `S{row}` pour Stake et
`Q{row}` comme cellule trigger commune. La valeur ecrite dans `Q{row}`
distingue `BACK`, `LAY`, `BACKSP` et `LAYSP`. Le script affiche la valeur
actuelle de cette cellule pour chaque runner et ne possede aucun chemin
d'ecriture Excel.

Pour previsualiser le nettoyage des cellules trigger non vides sur les lignes
runners WIN et PLACE:

```powershell
python scripts\clear_gruss_trigger_cells.py
```

Pour effectuer le nettoyage:

```powershell
$env:DOGBOT_GRUSS_CLEAR_TRIGGERS="true"
python scripts\clear_gruss_trigger_cells.py
python scripts\inspect_gruss_trigger_cells.py
```

Le nettoyeur est limite materiellement aux cellules `Q{row}` des runners
detectes en colonne A. Il ne peut pas modifier R/S et n'ecrit jamais de
commande `BACK`, `LAY`, `BACKSP` ou `LAYSP`. Les previews et nettoyages sont
journalises dans `data/gruss_trigger_clear_attempts.csv`.

Apres un trigger reel effectivement ecrit, le provider relit uniquement cette
cellule puis l'efface seulement si sa valeur correspond encore exactement au
trigger ecrit. Le delai de relecture est configurable avec
`DOGBOT_GRUSS_TRIGGER_CLEAR_DELAY_MS` et vaut `1500` ms par defaut. Pour les
tests reels, utiliser par defaut:

```powershell
$env:DOGBOT_GRUSS_TRIGGER_CLEAR_DELAY_MS="1500"
```

Le delai peut etre augmente a `2000` ou `3000` ms si Gruss a besoin de plus de
temps pour lire le trigger. Le nettoyage reste actif et ne s'applique jamais
en preview ni en write-no-trigger.

Toute ecriture reelle est maintenant relue immediatement dans R/S/Q. Un ordre
n'est journalise `GRUSS_REAL_WRITTEN` et compte dans la limite reelle que si
les trois valeurs relues correspondent aux valeurs ecrites. Sinon, le statut
est `GRUSS_WRITE_FAILED` avec `post_write_verification_failed`.

Pour rendre Q explicitement visible pendant le delai lors d'un real-test:

```powershell
$env:DOGBOT_GRUSS_HOLD_TRIGGER_FOR_VISUAL_TEST="true"
```

Cette option est refusee hors `DOGBOT_GRUSS_REAL_TEST_MODE=true`. Elle ne
desactive jamais le nettoyage: Q est efface apres
`DOGBOT_GRUSS_TRIGGER_CLEAR_DELAY_MS`.
