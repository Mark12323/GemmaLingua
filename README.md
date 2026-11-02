# GemmaLingua

**Fine-tuning Gemma 4 for Igbo, a low-resource language.**

GemmaLingua adapts Google's [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it) instruction-tuned model (`google/gemma-4-E4B-it`) to understand and generate [Igbo](https://en.wikipedia.org/wiki/Igbo_language), a language spoken by tens of millions of people in Nigeria.

The model is fine-tuned in three chained stages: language modelling, English-to-Igbo translation, and span infilling. Each stage trains a LoRA adapter on top of the previous one. The pipeline is language-agnostic: the language is defined in a single block of `config/pipeline.yaml`, so it can be retargeted at other Nigerian languages.

---

## Table of Contents

- [Why Igbo](#why-igbo)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Adapting to Another Language](#adapting-to-another-language)
- [Training Results](#training-results)
- [Contributing](#contributing)

---

## Why Igbo

Igbo is a low-resource language: there is far less digital text and far fewer public benchmarks for it than for languages like English. A general-purpose model may only see Igbo incidentally during pretraining.

GemmaLingua fine-tunes a capable base model directly on Igbo text instead, and measures whether that improves on the base model (see [Evaluation](#evaluation)).

---

## How It Works

Training runs in three sequential stages. Each stage produces a LoRA adapter, and the next stage starts from the previous adapter, so the final adapter contains all three stages.

```
google/gemma-4-E4B-it  (base)
        |
        v  Stage 1: Language modelling
igbo-gemma4-lm-finetuned            (base + LM)
        |
        v  Stage 2: English -> Igbo translation
igbo-gemma4-finetuned               (base + LM + translation)
        |
        v  Stage 3: Span infilling
igbo-gemma4-masking-finetuned       (base + LM + translation + masking)  <- final model
```

### Stage 1: Language modelling
Trains on raw Igbo text with a causal language modelling objective. There is no prompt: each line of text is a training example.

### Stage 2: Translation
Trains on English-Igbo sentence pairs using this prompt format:

```
Translate English to Igbo.
English: {src}
Igbo: {tgt}
```

### Stage 3: Span infilling
Trains on Igbo sentences with blanked spans, using this prompt format:

```
Fill in the blanks in the Igbo sentence.
Igbo: {masked}
Answer: {answer}
```

The blanks are derived from the Igbo (`ig`) column of the translation CSV. By default 15% of words are blanked per sentence, in spans averaging 3 words, with at most 5 blanks per sentence.

Each blank appears as `___` in the sentence. When a sentence has several blanks, the answers are joined with ` | ` in left-to-right order. Sentences shorter than 4 words are skipped, duplicate sentences are dropped, and masking is seeded (seed 42 by default) so the same data always produces the same blanks.

### Training details
- **Raw text prompts.** Although the base model is instruction-tuned, training uses plain text strings like the ones above, not the chat template.
- **LoRA.** Rank 16, alpha 32, dropout 0.05, applied to all linear layers (vision and audio towers are excluded).
- **4-bit quantization.** The base model is loaded in 4-bit (bitsandbytes) with bfloat16 compute.

---

## Project Structure

```
GemmaLingua/
├── config/
│   └── pipeline.yaml            # Pipeline stages, model, data paths, hyperparameters, language definition
├── python/
│   ├── cache_model.py           # Download and cache the base model locally
│   ├── download_translation.py  # Download an English-Igbo dataset from Hugging Face
│   ├── train_lm.py              # Stage 1
│   ├── train_translation.py     # Stage 2
│   ├── train_masking.py         # Stage 3
│   ├── eval_lm.py               # Evaluate language modelling
│   ├── eval_translation.py      # Evaluate translation
│   ├── eval_masking.py          # Evaluate span infilling
│   ├── requirements.txt         # Python dependencies
│   └── utils/
│       ├── data.py              # Dataset loading
│       └── model.py             # Model loading, LoRA config, adapter chaining
├── scripts/
│   ├── setup.sh                 # Create the venv, install dependencies, create data/ and models/ folders
│   └── run_pipeline.sh          # Build the Java orchestrator and run the full pipeline
├── src/                         # Java source for the pipeline orchestrator
├── pom.xml, mvnw, .mvn/         # Maven build files for the Java project
└── EVAL_PLAN.md                 # Evaluation methodology
```

The `data/`, `models/` and `logs/` directories are not part of the repository. `scripts/setup.sh` creates them (`data/raw`, `data/processed`, `models/base`, `models/adapters`, `logs`). The pipeline reads data from `./data/raw` and writes the cached base model to `./models/base` and trained adapters to `./models/adapters`.

Some files (`setup.sh`, the Java jar name, and `EVAL_PLAN.md`) refer to the project as "gemTun".

---

## Requirements

- An NVIDIA GPU with CUDA. Training loads the model in 4-bit with bitsandbytes and uses bf16 compute. The initial training run used a GPU with 16 GB of VRAM. The project has not documented a minimum.
- Python with the packages in `python/requirements.txt`: `torch`, `transformers`, `trl`, `peft`, `accelerate`, `bitsandbytes`, `datasets`, `pyarrow`, `pandas`, `evaluate`, and `sacrebleu`.
- A [Hugging Face](https://huggingface.co) account with access to `google/gemma-4-E4B-it`. Gemma models may require accepting Google's license terms before download.
- To run the full pipeline: JDK 21 (set in `pom.xml`) with `java` on your PATH. `./mvnw` runs the Maven wrapper jar at `.mvn/wrapper/maven-wrapper.jar` and stops with an error if that file is missing.
- The scripts in `scripts/` are bash scripts that expect a Linux-style virtual environment (`venv/bin/activate`). On Windows, run them from WSL, or follow the manual steps below.

---

## Setup

**1. Clone the repository:**
```bash
git clone https://github.com/Mark12323/GemmaLingua.git
cd GemmaLingua
```

**2. Create a virtual environment and install dependencies:**

On Linux, macOS or WSL, `./scripts/setup.sh` does this for you and also creates the `data/` and `models/` folders. To do it manually:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r python/requirements.txt
```

**3. Set your Hugging Face token:**
```bash
export HF_TOKEN=your_token_here          # Linux / macOS
$env:HF_TOKEN = "your_token_here"        # Windows PowerShell
```

**4. Cache the base model locally** (the script skips the download if the model is already cached):
```bash
python python/cache_model.py \
  --model-id google/gemma-4-E4B-it \
  --cache-dir ./models/base
```

---

## Data Preparation

### Stage 1: Igbo text

Prepare one or more `.txt` or `.csv` files of Igbo text. `train_lm.py` takes file paths through `--data`. It does not accept a directory.

Every non-empty line becomes one training example. CSV files are read as plain text rather than parsed as tables, so a header row would be trained on as if it were a sentence. Use a `.txt` file, or a CSV without a header.

When running the full pipeline, `config/pipeline.yaml` sets `data.lm_dir` (default `data/raw/monolingual`). Every `.txt` and `.csv` file in that directory is picked up automatically.

### Stages 2 and 3: English-Igbo pairs

Provide a CSV with an `en` column (English) and an `ig` column (Igbo):

```csv
en,ig
"<English sentence>","<Igbo translation>"
```

The default path in `config/pipeline.yaml` is `data/raw/igbo_translation_train.csv`. Rows with an empty value in either column are dropped, and when several files are given, duplicate rows are removed after merging.

To download a public dataset, run:
```bash
python python/download_translation.py
```

By default this downloads `iamwille/igbo-translation` from Hugging Face into `./data/raw` with the file prefix `igbo_translation`. Use `--dataset`, `--output`, and `--prefix` to change these.

### Hold out evaluation data first

Create a held-out split before training and keep it out of every training file. See [`EVAL_PLAN.md`](EVAL_PLAN.md) for how to split the data.

---

## Training

Run the commands from the repository root. Each stage saves its adapter to the `--output` directory.

### Stage 1: Language modelling
```bash
python python/train_lm.py \
  --data <igbo-text-file> [<more files> ...] \
  --output ./models/adapters/igbo-gemma4-lm-finetuned
```

### Stage 2: Translation
```bash
python python/train_translation.py \
  --data data/raw/igbo_translation_train.csv \
  --init-adapter ./models/adapters/igbo-gemma4-lm-finetuned \
  --output ./models/adapters/igbo-gemma4-finetuned
```

### Stage 3: Span infilling
```bash
python python/train_masking.py \
  --data data/raw/igbo_translation_train.csv \
  --init-adapter ./models/adapters/igbo-gemma4-finetuned \
  --output ./models/adapters/igbo-gemma4-masking-finetuned
```

`--init-adapter` continues training from a previous stage's adapter. This is what chains the stages together. `train_lm.py` has no such flag because Stage 1 starts from the base model.

### Run all stages with the pipeline orchestrator

On Linux, macOS or WSL, a Java orchestrator runs the whole pipeline from `config/pipeline.yaml`:

```bash
./scripts/setup.sh                 # once: creates the venv and folders
# edit config/pipeline.yaml so the data paths point to your files
export HF_TOKEN=your_token_here
./scripts/run_pipeline.sh
```

`run_pipeline.sh` activates the virtual environment, builds the Java project with `./mvnw clean package`, then runs `target/gemTun-1.0.0-jar-with-dependencies.jar`. Any arguments you pass to the script are passed on to the jar. The stages are listed in `config/pipeline.yaml`: `cache-base-model`, `validate-data`, `train-language-model`, `train-translation`, `train-masking`, and `export-model`.

### Options

Every training script accepts the same core options. The command-line defaults are:

| Option | Default |
|---|---|
| `--epochs` | 3 |
| `--batch-size` | 2 |
| `--grad-accum` | 4 |
| `--lr` | 2e-4 |
| `--lora-r` / `--lora-alpha` / `--lora-dropout` | 16 / 32 / 0.05 |
| `--logging-steps` | 50 |
| `--save-strategy` | epoch |

`config/pipeline.yaml` lists batch size 1 with gradient accumulation 8. Both combinations give an effective batch size of 8. `train_masking.py` also accepts `--mask-rate` (0.15), `--mean-span` (3), and `--max-spans` (5). Run any script with `--help` for the full list.

### Reported run times

From the training logs of the initial run, which used a GPU with 16 GB of VRAM (the GPU model is not recorded in this repository):

| Stage | Runtime |
|---|---|
| Language modelling | ~14.5 hours |
| Translation | ~2.4 hours |
| Span infilling | ~2.3 hours |

---

## Evaluation

Three scripts evaluate the models on held-out data: `python/eval_lm.py` (language modelling), `python/eval_translation.py` (translation), and `python/eval_masking.py` (span infilling).

The evaluation methodology, metrics and fairness rules are described in [`EVAL_PLAN.md`](EVAL_PLAN.md).

---

## Adapting to Another Language

For the full pipeline, change the four fields in the `language` block of `config/pipeline.yaml`. They drive the prompt templates and the CSV column names:

```yaml
language:
  source_code: "en"        # CSV column holding the source text
  source_name: "English"   # name used in the prompt
  target_code: "yo"        # CSV column holding the target text
  target_name: "Yoruba"    # name used in the prompt
```

When running scripts individually, pass the equivalent flags instead: `--source-col`, `--source-name`, `--target-col`, and `--target-name` (accepted by `train_translation.py`, `train_masking.py`, `eval_translation.py`, and `eval_masking.py`).

---

## Training Results

Final token accuracy from the initial training run:

| Stage | Adapter | Token accuracy | Average train loss |
|---|---|---|---|
| Language modelling | `igbo-gemma4-lm-finetuned` | 0.5989 | 2.105 |
| Translation | `igbo-gemma4-finetuned` | 0.7705 | 1.42 |
| Span infilling | `igbo-gemma4-masking-finetuned` | 0.7595 | 1.49 |

Token accuracy is top-1 next-token accuracy. Loss values are not comparable across stages, because each stage optimises a different objective. For how to interpret these numbers and how the models are compared against the base model, see [`EVAL_PLAN.md`](EVAL_PLAN.md).

---

## Contributing

Contributions are welcome, especially:

- Igbo text data: monolingual corpora, parallel translation pairs, or pointers to public datasets
- Evaluation results from running the eval scripts
- Retargeting the pipeline to other Nigerian languages
- Documentation improvements

Please open an issue before starting significant work so efforts are not duplicated. Pull requests should target the `staging` branch.
