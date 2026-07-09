# Known Issues

Key limitations and known issues of the current system.

## 1. Mathematical Logic of Small Models
- **Symptoms:** Simple equations (e.g. `5 + 3`) can yield incorrect arithmetic outputs (e.g., `55`) when running under the small 4.7M parameter locally-trained checkpoints.
- **Cause:** The model parameter capacity (vocab size 1,029 or 1,481, embedding dimension 256, 6 layers) is too small to build generalizable mathematical reasoning.
- **Workaround:** Train a larger model (e.g., more layers or larger embedding dimension) on a larger pretraining dataset corpus to build stronger arithmetic generalizations.

## 2. Windows Unicode Encoding Output Errors
- **Symptoms:** running scripts on standard Windows terminal consoles can throw `UnicodeEncodeError: 'charmap' codec can't encode character...`
- **Cause:** Python output functions default to standard Windows codepages (like CP1252) which lack Unicode glyphs (like arrow or checkmark symbols).
- **Fix:** Restrict terminal logging outputs to standard ASCII character sets.

