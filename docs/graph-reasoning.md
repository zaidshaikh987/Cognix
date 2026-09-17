# Graph Reasoning

COGNIX supports representing multi-agent networks as graphs, where edges indicate trust or communicative links between agents.

| Model | Graph reasoning | Epistemic prior | Intended role |
|---|---:|---:|---|
| **NoGraph** | No | No | Non-graph baseline for simple weighted fusion. |
| **StandardGAT** | Yes | No | Standard graph attention modelling structural relationships. |
| **EpistemicGAT** | Yes | Yes | Uncertainty-aware graph attention with monotonic suppression. |

## EpistemicGAT

The core innovation in COGNIX's graph module is the `EpistemicGAT` (Epistemic Graph Attention Network).

In a standard GAT, agent relationships are represented through learned graph attention weights $e_{ij}$. However, standard GATs are blind to out-of-distribution (OOD) model ignorance. 

`EpistemicGAT` explicitly injects **epistemic uncertainty** as a prior into the attention calculation. When an agent experiences high epistemic uncertainty (meaning it does not know what it is looking at), the prior monotonically suppresses that agent's outgoing attention weights.

Mathematically, this is implemented as an additive log prior on the attention logits:

$$ e'_{ij} = e_{ij} + \log(p_{ij}) $$

Where the prior $p_{ij}$ is defined as a function of the sender's epistemic uncertainty $\sigma_{e,j}$:

$$ p_{ij} = \frac{1}{1 + \sigma_{e,j}} $$

This mathematically guarantees that as $\sigma_{e,j} \to \infty$, the attention weight $e'_{ij} \to -\infty$, effectively silencing the uncertain agent in the softmax output.

### Usage

```python
from cognix import EpistemicGAT

gnn = EpistemicGAT(
    num_layers=2,
    input_dim=3,
    hidden_dim=8,
    output_dim=4,
    use_epistemic_prior=True
)
```
