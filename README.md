# Evolutionary-Algorithms

ISE Project for James Patten


```
cd GE_House_Price
```
```
pip install -r requirements.txt
```
```
python main.py
```


```
GE_House_Price/
│
├── data/
│   └── houses.csv
│
├── grammar/
│   └── houseprice.bnf
│
├── src/
│   ├── data_preprocessing.py
│   ├── grammar.py
│   ├── ge_main.py
│   ├── evaluation.py         
│   └── visualisation.py
│
├── results/
│   └── ...
│
├── requirements.txt
└── main.py              
```




## ToDo/ideas

- [ ] Change crossover from single-point to multi-point or uniform
- [ ] Implement mutation and dynamic mutation rate based on fitness (stretch goal)
- [ ] Implement max depth of tree, instead of forcing a terminal option to be chosen
- [ ] Ensure mutation can decrease tree depth
- [ ] Add the following as operators for expression generation:
    - [ ] Arithmetic: `+`, `-`, `*`, `/`, `-x`
    - [ ] Trigonometric: `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`
    - [ ] Exponentials: `e^x`, `e^-x`, `ln(x)`
    - [ ] Power: `x^2`, `x^3`, `e^x`, `1/x`
