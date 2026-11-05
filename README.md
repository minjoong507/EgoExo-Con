# EgoExo-Con: Exploring View-Invariant Video Temporal Understanding

[![arXiv](https://img.shields.io/badge/arXiv-2510.26113-b31b1b.svg)](https://arxiv.org/abs/2510.26113) 
<a href='https://huggingface.co/datasets/mjjung/EgoExo-Con'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-EgoExo--Con-yellow'></a>
<a href='https://huggingface.co/mjjung/View-GRPO'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-View--GRPO-blue'></a>


**EgoExo-Con** is a benchmark for measuring the *ego–exo consistency* of Video Large
Language Models (Video-LLMs): given the same event captured from both an
**exocentric** (third-person) and an **egocentric** (first-person) viewpoint, a model
should answer consistently. The benchmark covers two temporal tasks:

- **`verif`** — *temporal verification*: decide whether a queried event occurs within a
  given time window (`Yes` / `No`).
- **`grounding`** — *temporal grounding*: predict the start and end time of a queried event.

The released baseline evaluates [Qwen2.5-VL](https://huggingface.co/Qwen).

## Repository layout

```
run.py                              # Entry point: run inference + evaluation
run_eval.sh                         # Re-evaluate already-saved predictions
baselines/qwen_2_5_vl.py            # Qwen2.5-VL model wrapper
prompts/qwen.py                     # Task prompts for Qwen
task/egoexo_consistency_batch.py    # Batched inference loop
task/eval.py                        # Scoring (GPT judge for verification, IoU for grounding)
utils/cons_utils.py                 # Options, I/O, logging, IoU, question generation
utils/vision_process.py             # Video frame sampling (vendored from qwen-vl-utils)
data/EgoExo-Con.json                # Benchmark annotations
```

## Installation

```bash
pip install -r requirements.txt
```

## Data

`data/EgoExo-Con.json` is a list of samples, each shaped like:

```json
{
  "exo_video_path": "V6RK6.mp4",
  "ego_video_path": "V6RK6EGO.mp4",
  "temporal_query": "A person is standing by a washing machine and holding a pillow.",
  "misaligned_query": "A person is holding a small blanket while sitting on a bed in a dark room.",
  "length": 26.79,
  "source": "CharadesEgo",
  "timestamp": [14.9, 26.79]
}
```

`source` is one of `CharadesEgo`, `EgoExo4D`, `Lemma`. EgoExo-Con reuses the videos from
these three public datasets, so only the annotations are distributed here — you need to
download the raw videos yourself (see below).

## Datasets

Videos are loaded from `--video_root` (default: `video_datasets/`) using the pattern:

```
<video_root>/<source>/<id>.mp4       # exocentric (third-person) view
<video_root>/<source>/<id>EGO.mp4    # egocentric (first-person) view
```

So for each annotation, the paired ego/exo clips live side by side in the same `source`
folder, with the egocentric file suffixed by `EGO`. The expected layout is:

```
video_datasets/
├── CharadesEgo/
│   ├── V6RK6.mp4              # exo
│   ├── V6RK6EGO.mp4           # ego
│   └── ...
├── EgoExo4D/
│   ├── b60db7b1-c623-4f04-8c63-695c24b4648b_1.mp4
│   ├── b60db7b1-c623-4f04-8c63-695c24b4648b_1EGO.mp4
│   └── ...
└── Lemma/
    ├── 28k-12-11-1.mp4
    ├── 28k-12-11-1EGO.mp4
    └── ...
```

The exact clip ids to prepare for each split are listed in `data/EgoExo-Con.json`
(`exo_video_path` / `ego_video_path`). EgoExo-Con uses **601** CharadesEgo, **1596**
EgoExo4D, and **179** Lemma query samples.

### Obtaining the source videos

Each split is built on top of an existing released dataset. Download the originals from
their official sources (license/terms apply), trim each annotated clip to the start/end
times referenced by EgoExo-Con, and save the ego/exo pair under the matching `source`
folder using the `<id>.mp4` / `<id>EGO.mp4` naming above.

| Split | Source dataset | Official link |
|-------|----------------|---------------|
| `CharadesEgo` | Charades-Ego | https://prior.allenai.org/projects/charades-ego |
| `EgoExo4D`    | Ego-Exo4D     | https://ego-exo4d-data.org |
| `Lemma`       | LEMMA         | https://sites.google.com/view/lemma-activity |

> **Note**
> - **Charades-Ego** ships paired third-person (`<id>.mp4`) and first-person (`<id>EGO.mp4`)
>   recordings out of the box; rename/copy them into `video_datasets/CharadesEgo/`.
> - **Ego-Exo4D** and **LEMMA** provide time-synchronized egocentric and exocentric
>   captures of the same takes; export the egocentric stream as `<id>EGO.mp4` and the
>   exocentric stream as `<id>.mp4`.
>
> A clip's `length` (seconds) and ground-truth `timestamp` are stored in the annotation,
> so you can verify each prepared video matches the expected duration.

## Usage

Run inference and evaluation:

```bash
python run.py \
    --model_type qwen2_5 \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --task grounding \
    --video_root /path/to/videos \
    --eval_by_split
```

Use `--task verif` for temporal verification and `--task grounding` for temporal
grounding. Add `--debug` for a quick smoke test on a couple of batches. Predictions and
scores are written under `<task>_results/<model_type>/<exp_id_or_timestamp>/`.

To re-score existing predictions without re-running the model:

```bash
bash run_eval.sh --test_path <output_dir> --eval_by_split
```

### OpenAI API key

Verification scoring uses `gpt-4o-mini` as a judge for free-form answers. Provide your
key via the environment (never commit it):

```bash
export OPENAI_API_KEY=sk-...
```

## Citation
If you find our work useful, please cite:
```aiignore
@inproceedings{jung2026egoexo,
  title={Egoexo-con: Exploring view-invariant video temporal understanding},
  author={Jung, Minjoon and Xiao, Junbin and Kim, Junghyun and Zhang, Byoung-Tak and Yao, Angela},
  booktitle={European Conference on Computer Vision},
  pages={355--373},
  year={2026},
  organization={Springer}
}
```
