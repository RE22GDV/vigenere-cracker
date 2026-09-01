# Architecture

How the program is put together and why. All diagrams render natively on GitHub.

- [1. End-to-end pipeline](#1-end-to-end-pipeline)
- [2. Choosing a strategy](#2-choosing-a-strategy)
- [3. Frequency analysis](#3-frequency-analysis)
- [4. Brute force](#4-brute-force)
- [5. Two-stage scoring](#5-two-stage-scoring)
- [6. Execution and parallelism](#6-execution-and-parallelism)
- [7. Language model construction](#7-language-model-construction)
- [8. Module dependencies](#8-module-dependencies)

---

## 1. End-to-end pipeline

From ciphertext to a ranked list of candidates.

```mermaid
flowchart TD
    A["Ciphertext + language"] --> B["Alphabet variants<br/>en-26 / ru-32 / ru-33 / uk-33 / uk-32"]
    B --> C["Letter indices + layout template<br/>spacing and punctuation kept aside"]
    C --> D["Key-length hints<br/>index of coincidence, Kasiski"]
    C --> E{"Strategy<br/>see diagram 2"}

    E --> F1["Brute force<br/>GPU or CPU pool"]
    E --> F2["Frequency analysis<br/>key is computed"]
    E --> F3["Dictionary attack"]
    E --> F4["Hill climbing"]

    F1 --> G["Candidate pool<br/>score + key, per alphabet"]
    F2 --> G
    F3 --> G
    F4 --> G

    G --> H["Stage 1: quadgram re-scoring<br/>all candidates"]
    H --> I["Stage 2: dictionary coverage<br/>top 600 only"]
    I --> J["Key-length penalty<br/>ln M per key letter"]
    J --> K["Merge across alphabets, sort"]
    K --> L["Ranked candidates<br/>key, alphabet, decryption"]

    D -.advisory only.-> L
```

The layout template is what lets the result come back as `find me at the old stone`
rather than `findmeattheoldstone`: only letters are enciphered, everything else is
restored in place.

---

## 2. Choosing a strategy

`Auto` mode does not try everything blindly — the useful method depends on how
much ciphertext there is.

```mermaid
flowchart TD
    A["Ciphertext, n letters"] --> B{"n < 60?"}

    B -->|"yes: short text"| C["Statistics per column are too thin<br/>for frequency analysis,<br/>but short keys are fully enumerable"]
    C --> C1["1. Brute force, key 1..5 GPU / 1..4 CPU"]
    C1 --> C2["2. Dictionary, frequent words"]
    C2 --> C3["3. Frequency analysis as a long shot"]

    B -->|"no: long text"| D["The key follows from the text itself,<br/>brute force is pointless"]
    D --> D1["1. Frequency analysis, key 1..min(28, n/4)"]
    D1 --> D2["2. Dictionary, frequent words"]
    D2 --> D3["3. Short brute force as a safety net"]

    C3 --> E["Common candidate pool"]
    D3 --> E
```

Phases are ordered so the most likely winner runs first: partial results appear
in the table while the rest is still running, and `Stop` at any moment leaves a
usable answer.

---

## 3. Frequency analysis

The method that makes key length irrelevant. Work grows as **L**, not as **32^L**.

```mermaid
flowchart TD
    A["Assume key length L"] --> B["Split text into L columns<br/>column i = letters i, i+L, i+2L, ..."]
    B --> C["Inside a column the cipher<br/>is a plain Caesar shift"]
    C --> D["For each of M possible key letters:<br/>decrypt the column, correlate<br/>letter frequencies with the language"]
    D --> E["Rank the candidate letters<br/>per column"]
    E --> F["Seed key = best letter of every column"]

    F --> G["Hill climb on trigrams:<br/>try all M letters in each position,<br/>keep any improvement"]
    G --> H{"Improved?"}
    H -->|yes| G
    H -->|no| I{"Seeds left?"}

    I -->|"yes, 65% of the time"| J["Perturb the incumbent:<br/>replace 1..L/3 positions with<br/>other high-ranking letters"]
    I -->|"yes, otherwise"| K["Fresh guess from the<br/>top-3 letters per column"]
    J --> G
    K --> G

    I -->|no| L["Collapse to minimal period<br/>abcabc becomes abc"]
    L --> M["Best keys for this L"]
```

**Why the perturbation loop exists.** With a 12-letter key on an 80-letter text
each column holds only 6–8 letters; the frequency signal is close to noise and
plain hill climbing settles one or two letters away from the answer — it produced
`cryptogrbphz` instead of `cryptography`. Diagnosis showed the true key scored
*much* higher (−18 versus −124 nats), so the model was right and the search was
at fault. Iterated local search fixed every case of this kind.

---

## 4. Brute force

Exact, and the only option when there is too little text for statistics.

```mermaid
flowchart LR
    A["Key length L"] --> B["Key space M^L<br/>enumerated as integers"]
    B --> C["Split into work units"]

    subgraph GPU["GPU path, CUDA"]
        G1["Batch of ~1M key ordinals"] --> G2["Decode to base-M digits"]
        G2 --> G3["Expand to text length,<br/>decrypt whole batch"]
        G3 --> G4["Gather trigram scores, sum"]
        G4 --> G5["top-k stays in video memory"]
        G5 --> G6{"Out of memory?"}
        G6 -->|yes| G7["Halve the batch, retry"]
        G7 --> G1
        G6 -->|no| G8["Next batch"]
        G8 --> G1
    end

    subgraph CPU["CPU path, process pool"]
        P1["Work unit = range of ordinals"] --> P2["Vectorised numpy batch<br/>of 65536 keys"]
        P2 --> P3["Local top-60 heap"]
    end

    C --> GPU
    C --> CPU
    G5 --> D["Candidate pool"]
    P3 --> D
```

Keeping the leaders in video memory matters: moving the top of every batch back
into Python cost more than the search itself — 81 s versus 11.6 s for a 6-letter
key.

---

## 5. Two-stage scoring

Search and ranking use deliberately different metrics.

```mermaid
flowchart TD
    A["Billions of candidates"] --> B["Trigram log-probability<br/>table M^3, fits in cache"]
    B --> C["Keep a few thousand leaders"]

    C --> D["Quadgram log-probability<br/>table M^4, much stricter"]
    D --> E["Sort by quadgram score<br/>minus key-length penalty"]
    E --> F["Top 600 only:<br/>dictionary coverage by DP"]

    F --> G["final = quadgrams<br/>+ 6.0 x coverage x n<br/>- ln M x key length"]
    G --> H["Ranked list"]

    style B fill:#e8f0fe
    style D fill:#e8f5e9
    style F fill:#fff4e5
```

Each term earns its place:

| Term | Why it is there |
|---|---|
| Trigrams | cheap enough for billions of candidates |
| Quadgrams | separates real language from "almost language" |
| Dictionary coverage | decisive on short texts, where letter statistics are thin |
| Key-length penalty | without it the longest key always wins — with a key as long as the text, *any* plaintext is reachable |

Dictionary coverage is limited to 600 leaders because its cost grows with text
length: unrestricted, it took 16 s out of a 25 s run on a 381-letter text.

```mermaid
flowchart LR
    A["Decryption without spaces<br/>meetmeatmidnight"] --> B["Dynamic programming:<br/>best split into real words"]
    B --> C["Words shorter than 3 letters ignored<br/>they match by chance"]
    C --> D["Longer words weigh more, L^1.4"]
    D --> E["coverage = sum of weights<br/>normalised by n^1.4"]
```

---

## 6. Execution and parallelism

```mermaid
flowchart TD
    UI["GUI / CLI thread"] -->|config| ENG["Engine thread"]
    ENG -->|events: progress, interim, results| UI

    ENG --> PLAN["Work plan:<br/>alphabet x phase x key length"]

    PLAN --> GPUP["GPU phases<br/>run in the engine thread"]
    PLAN --> POOL["CPU phases"]

    subgraph POOLBOX["One process pool per run"]
        W1["Worker 1"]
        W2["Worker 2"]
        WN["Worker N"]
    end

    POOL --> POOLBOX
    POOLBOX -->|progress queue| ENG
    POOLBOX -->|results| ENG

    SHARED["Shared via files, not IPC:<br/>trigram table .npy<br/>dictionary matrices .npz"] -.-> POOLBOX
```

Three decisions that came out of profiling:

1. **One pool per run**, serving every alphabet and every phase. Starting 32
   processes costs about 4.5 s on Windows and should be paid once, not per
   alphabet.
2. **Heavy tables go through files.** Workers memory-load the trigram table and
   the dictionary matrices themselves; passing them as arguments would move
   megabytes per task.
3. **The dictionary is converted once, in the parent.** Letting each of 32
   workers parse millions of words took longer than the attack itself —
   15 s versus 1.6 s.

---

## 7. Language model construction

```mermaid
flowchart TD
    A["corpus_LANG.txt<br/>words_LANG.txt"] --> B["Lowercase, encode to a<br/>single-byte encoding<br/>cp1251 or latin-1"]
    B --> C["One 256-entry lookup table<br/>byte to letter index, or separator"]
    C --> D["Count n-grams with bincount<br/>inside words, weight 1.0"]
    C --> E["Count n-grams across word breaks,<br/>weight 0.35 - ciphertext has no spaces"]

    D --> F["Unigrams, bigrams,<br/>trigrams, quadgrams"]
    E --> F
    F --> G["Jelinek-Mercer interpolation<br/>with back-off to shorter contexts"]
    G --> H["log-probability tables"]
    H --> I["Cache to model_LANG_VARIANT_HASH.npz<br/>keyed on source file size and mtime"]

    A --> J["Word set for coverage scoring<br/>folds applied: yo to e, ge to g"]
```

The single-byte trick is what makes a 73 MB corpus process in 7.5 s: turning text
into letter indices is one array lookup instead of a per-character Python loop.
Letter frequencies for the frequency analysis are then read back out of this
model, so no language needs a hand-written frequency table.

---

## 8. Module dependencies

```mermaid
flowchart BT
    paths["paths.py<br/>data directory"]
    langs["langs.py<br/>alphabets, built-in samples"]
    ru["ru_data.py<br/>Russian sample"]
    i18n["i18n.py<br/>interface strings"]
    core["core.py<br/>alphabets, cipher, model,<br/>IC, Kasiski, correlation"]
    attack["attack.py<br/>engine and attacks"]
    calib["calibrate.py<br/>throughput measurement"]
    gui["gui.py"]
    cli["cli.py"]
    dl["download_data.py"]

    ru --> langs
    langs --> core
    paths --> core
    core --> attack
    i18n --> attack
    attack --> calib
    attack --> gui
    attack --> cli
    calib --> gui
    core --> dl
    paths --> dl
```

`core` knows nothing about attacks, `attack` knows nothing about the interface,
and adding a language touches only `langs.py` and `download_data.py`.
