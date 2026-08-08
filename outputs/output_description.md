# Preliminary output description

The pipeline selected **10** representative CFPB complaint narratives and used **google/flan-t5-small** with deterministic beam-search decoding. Each prompt included the cleaned complaint, a provisional transparent archetype, extracted red flags, and a rule-based risk level. The outputs are preliminary and require human review because the generator is not an independent fact checker and the provisional archetypes are not the final trained classifier labels.

Run mode: **prepare only**.
