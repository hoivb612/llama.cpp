# llama-mm-embed

Image <-> text affinity with a multimodal embedding model such as EmbeddingGemma 2.

Images and texts are embedded into the same vector space (mean pooling, L2-normalized), then
compared with cosine similarity. Each input is decoded in a single ubatch with its text tokens and
image embeddings mixed, which is required because the model uses bidirectional attention.

## Usage

```
llama-mm-embed -m embeddinggemma-2-Q8_0.gguf --mmproj mmproj-Q8_0.gguf \
    --image-dir photos/ \
    --text "sunset over a calm lake" --text "abstract blue curved shapes"
```

Inputs (repeatable, any mix): `--image FILE`, `--image-dir DIR`, `--text TEXT`, `--text-file FILE`.

Output:
- image x text similarity matrix (best match per row marked `*`)
- top-k texts for each image and top-k images for each text
- with `--all-pairs`, or when only one modality is given: image x image / text x text

Options:
- `--text-prefix STR`: task prefix added to every text (default `task: search result | query: `).
  Images never get a prefix. Use `--text-prefix ""` to disable.
- `--dim N`: Matryoshka truncation to 512/256/128 dims (re-normalized after truncation).
- `--image-max-tokens N`: cap vision tokens per image. A 1920x1080 image uses ~1100 tokens by
  default; `--image-max-tokens 280` is ~8x faster on CPU and usually still ranks correctly.
- `-c N`: context = batch = ubatch size; must hold the largest single input (default 4096).
- `--top-k N`, `-ngl N`, `-t N`, `-v`.

`--mmproj` is optional for text-only runs.

## Notes

- `run-tests.ps1 -ModelDir <dir>` runs the smoke tests (synthetic shapes, Windows wallpapers,
  capped image tokens, text only); `-Only shapes|photos|capped|text` runs one.
- Use BF16 or Q8_0 weights; the model card warns that F16 activations overflow.
- Similarities are relative: unrelated pairs typically score ~0.5-0.6, good matches ~0.75-0.8.
