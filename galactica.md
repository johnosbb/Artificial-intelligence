
# Galactica

The original promise of computing was to solve information overload in science. But classical computers were specialized for retrieval and storage, not pattern recognition. As a result, we've had an explosion of information but not of intelligence: the means to process it. Researchers are buried under a mass of papers, increasingly unable to distinguish between the meaningful and the inconsequential. Galactica aims to solve this problem.


# References

- [The Paper](https://galactica.org/static/paper.pdf)

```sh
pip install galai
```

```python
import galai as gal

model = gal.load_model("huge")
model.generate("The Transformer architecture [START_REF]")
# The Transformer architecture [START_REF] Attention is All you Need, Vaswani[END_REF] has been widely used in natural language processing.
```
