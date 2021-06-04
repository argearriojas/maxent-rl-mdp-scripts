# rl-scripts

## Install dependencies
```
conda create -n ulearn python
conda activate ulearn
conda install -c conda-forge numpy pandas gym matplotlib scipy joblib progress
```

## Running examples

```
# reproduce figure 2
python example_v_values.py
```
![Fig 2](fig2.png)

```
# reproduce figure 3
python example_shortest_path.py
```
![Fig 3](fig3.png)
```
# reproduce figure 4
python example_model_free.py
```
![Fig 4](fig4.png)

```
# reproduce figure S1
python example_irreducible_towards_reducible.py
```
![Fig S1](figS1.png)
