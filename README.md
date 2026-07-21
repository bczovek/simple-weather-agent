# Egyszerű időjárás agent

> Ez a felhasználói dokumentáció. Az implementáció belső felépítéséért,
> a LangGraph gráf működéséért és tervezési döntésekért lásd a
> [DEVELOPERS.md](DEVELOPERS.md) fájlt.

## Követelmények

- Python 3.11 vagy újabb
- Érvényes API kulcs egy OpenAI API-val kompatibilis LLM szolgáltatáshoz

## Környezeti változók

Az agent a `.env` fájlból olvassa be a konfigurációt (a `simple-weather-agent`
könyvtárban). Először másold le a mintafájlt:

```bash
cd simple-weather-agent
cp .env.example .env
```

majd töltsd ki a `.env` fájlt:

| Változó | Kötelező | Alapérték | Leírás |
|---|---|---|---|
| `OPENAI_API_KEY` | igen | – | Az LLM szolgáltatás API kulcsa. |
| `OPENAI_MODEL` | nem | `gpt-4o` | A használt modell neve. |
| `OPENAI_BASE_URL` | nem | OpenAI alapértelmezett végpontja | Más OpenAI-kompatibilis végpont (pl. Azure OpenAI, lokális LM Studio/Ollama, Groq) elérési útja. Ha üresen hagyod, az OpenAI alapértelmezett végpontját használja. |

> Megjegyzés: az `OPENAI_JUDGE_MODEL` változó csak az automatikus
> teszteléshez használatos, a
> normál futtatáshoz nem szükséges.

## Futtatás

Az agentet a `run.sh` szkripttel kell indítani, amely:

1. létrehoz egy lokális, a projekt könyvtárában lévő Python virtuális
   környezetet (`./.venv`), ha még nem létezik,
2. aktiválja azt,
3. telepíti a szükséges depedency-ket ebbe a virtuális környezetbe,
4. elindítja az interaktív agentet.

```bash
cd simple-weather-agent
./run.sh
```

Ezután egy interaktív parancssori bemenetre vár:

```
Enter a question about a city's current temperature, or type 'exit' to quit.
> Mekkora most a hőmérséklet Budapesten?
```

A kilépéshez írd be, hogy `exit` (vagy nyomj `Ctrl+D`-t).

### Parancssori paraméterek

| Paraméter | Kötelező | Leírás |
|---|---|---|
| `--verbose` / `-v` | nem | Kiírja az agent minden egyes reasoning (döntési) és tool-hívási lépését futás közben. |

Példák:

```bash
# Normál mód: csak a végső válasz jelenik meg
./run.sh

# Részletes mód: minden reasoning/tool-calling lépés látható
./run.sh --verbose
```

## Az agent viselkedése

- Csak városok **jelenlegi** hőmérsékletére vonatkozó kérdésekre válaszol
  (ideértve az ilyen hőmérsékletek összehasonlítását vagy egyszerű
  számításait, pl. "melyik város melegebb?").
- Múltbeli vagy jövőbeli (előrejelzett) hőmérsékletre, illetve bármilyen más
  témára vonatkozó kérdés esetén udvariasan elutasítja a választ, anélkül,
  hogy az adott témában bármilyen információt adna.
- Összetett (több várost vagy több altémát érintő) kérdéseknél csak a
  hőmérsékletre vonatkozó részt válaszolja meg, a többit elutasítja.

## További dokumentáció

Az implementáció belső felépítéséért (modulok, a LangGraph állapotgép
gráfja mermaid diagrammal, tervezési döntések, trade-off-ok) lásd a
[DEVELOPERS.md](DEVELOPERS.md) fejlesztői dokumentációt.

