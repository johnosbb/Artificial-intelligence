# Text Style Transfer Glossary

| Term | Definition |
| ----- | ----- |
| BART |  BART (Bidirectional and Auto-Regressive Transformers) is a transformer-based model introduced by Facebook AI Research (FAIR). It is designed for sequence-to-sequence tasks and has been used for various natural language processing (NLP) applications, including text summarization. BART is described as a denoising autoencoder because it is trained to reconstruct a corrupted or noisy version of its input. In the context of BART, this means taking a sequence of text, corrupting it in some way (e.g., by masking out certain words or segments), and training the model to generate the original, uncorrupted sequence. BART is pre-trained in a self-supervised fashion on 160GB of news, books, stories, and web text by corrupting input sentences with a noising function and then learning a model to reconstruct the original text. |
| Condition Sequencing | Conditional generation typically involves generating output sequences based on a given input and some form of conditioning information.  |
| Self Supervised Learning |  Self-supervised learning is a machine learning paradigm where models are trained to learn from unlabeled data without explicit supervision. In traditional supervised learning, models are trained on labeled datasets, where input data is paired with corresponding output labels. However, obtaining labeled data can be expensive and time-consuming. 
In self-supervised learning, the idea is to create a supervision signal from the data itself without requiring external labels. The model is trained to generate labels or representations from the input data in a way that captures meaningful information. This approach is particularly useful when labeled data is scarce or unavailable. |
| Transfer Learning | Transfer learning is a machine learning technique where a model trained on one task is repurposed for a different, but related, task. The idea is to leverage knowledge gained from solving a source task to improve learning on a target task. Instead of training a model from scratch for the target task, transfer learning initializes the model with knowledge obtained from a pre-trained model on a different task. |