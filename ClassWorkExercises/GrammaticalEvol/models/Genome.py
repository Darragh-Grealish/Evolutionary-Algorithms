import random

class Genome:
    def __init__(self):
        self.codons = []

    def initialize_codons():
        for i in range(int((random.random() * 10))):

            print(f"index {i}")
            self.codons.append(int(random.random() * 255))

