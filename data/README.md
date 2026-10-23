# Dataset format

Keep training datasets out of Git unless each item is licensed for redistribution and contains no private or sensitive information. The repository ignores local JSONL files by default.

## JSONL schema

One JSON object per line, with a `messages` array using chat roles:

```json
{"messages":[{"role":"user","content":"اشرح الفرق بين القائمة والطابور في بايثون مع مثال."},{"role":"assistant","content":"القائمة ..."}]}
```

Supported roles are `system`, `user`, and `assistant`. Each message must have non-empty string `content`. Include a final assistant response. The preparation script validates the structure and serializes the conversation using the base tokenizer's chat template during training.

## Recommended dataset mix

- Arabic programming explanations and code walkthroughs.
- Debugging tasks with a reproducible error and a verified fix.
- Arabic technical Q&A, including correct standard terminology.
- Code generation tasks with unit tests or expected behavior.
- Bilingual Arabic/English technical terms where useful.
- A small, held-out evaluation set that is never included in training.

## Quality and governance

For every dataset, record its source, license, collection date, transformation steps, and known limitations. Deduplicate near-identical examples, review generated answers, and avoid benchmarks in training. Do not include credentials, personal data, copyrighted material without permission, or evaluation answers that should remain held out. Prefer verified code examples and run tests where practical.
