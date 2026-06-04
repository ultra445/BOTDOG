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
