```sh
pip install galai
```

```python
import galai as gal

model = gal.load_model("huge")
model.generate("The Transformer architecture [START_REF]")
# The Transformer architecture [START_REF] Attention is All you Need, Vaswani[END_REF] has been widely used in natural language processing.
```
