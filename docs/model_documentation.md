# Model Documentation

## Selected generator

The pipeline uses `google/flan-t5-small`, an instruction-tuned encoder-decoder Transformer. The
task is a structured text-to-text mapping: complaint narrative plus fraud signals in, concise
analyst summary out.

## Inference design

- maximum input length: 512 tokens;
- maximum new tokens: 96;
- deterministic four-beam decoding;
- no sampling;
- batches of four on GPU by default;
- ten representative samples selected across common sub-products.

## Structured inputs

Red flags and risk levels are produced by transparent rules. A provisional archetype is also
assigned by ordered keyword patterns. These features support reproducibility and make the prompt
inspectable, but they are not substitutes for the planned trained classifier.

## Upgrade path

When a valid fine-tuned checkpoint is stored under `models/generator/`, the model runner uses it
automatically. The same input and output interfaces remain unchanged, so the zero-shot and
fine-tuned systems can be compared without rewriting the pipeline.
