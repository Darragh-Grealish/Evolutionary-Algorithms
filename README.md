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

Run Tests
```
pytest
# Or Verbose Details
pytest -v
```



Project Structure
```
GE_House_Price/
│
├── data/
│   └── houses.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── evalution.py
│   ├── ge_main.py
│   ├── genetic_operators.py 
│   ├── grammar.bnf         
│   ├── models.py         
│   ├── population.py         
│   └── visualisation.py
│
├── test/
│   ├── population_test.py      
│   └── evaluation_test.py
│
├── results/
│   └── ...
│
├── logs/
│   └── ...
│
├── config.json
│
├── requirements.txt
│
├── pytest.ini
│
└── main.py              
```

