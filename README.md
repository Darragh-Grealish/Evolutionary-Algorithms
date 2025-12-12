# Evolutionary-Algorithms

Create or Activate Virtual Environment
```
# create
python3 -m venv .venv

# activate
source .venv/bin/activate
```

Install Requirements
```
pip install -r requirements.txt
```

Run GE Model
```
python main.py
```




Project Structure
```
Evolutionary-Algorithms/
│
├── data/
│   └── houses.csv
│
├── grammar/
│   └── houseprice.bnf
│
├── src/
│   ├── config.py 
│   ├── data_preprocessing.py
│   ├── genome.py 
│   ├── grammar.py
│   ├── ge_main.py
│   ├── evaluation.py         
│   └── visualisation.py
│
├── results/
│   └── ...
│
├── requirements.txt
│
├── config.json
│
└── main.py              
```

