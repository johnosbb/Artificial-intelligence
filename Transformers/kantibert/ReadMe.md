## Files in output directory

- checkpoint-2672
- config.json
- merges.txt
- model.safetensors
- training_args.bin
- vocab.json

 ### Model

- model.safetensors: This file contains the model weights. In previous versions, this might have been a .bin file (e.g., pytorch_model.bin), but the .safetensors format is also used to store model weights safely and efficiently.

### Tokenizer

- vocab.json: This file contains the vocabulary of the tokenizer.
- merges.txt: This file contains the merge rules for the Byte-Pair Encoding (BPE) tokenizer.
- config.json: This configuration file includes details about the model architecture and tokenizer settings. It is necessary for both the model and tokenizer.
- The training_args.bin file contains the training arguments used during the training process, and the checkpoint-2672 directory is likely a checkpoint from a specific training step, containing its own set of model weights and other relevant files.