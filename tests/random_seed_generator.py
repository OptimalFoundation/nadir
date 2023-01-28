import os
import numpy as np

if __name__ == '__main__':
    seeds = np.random.randint(10000000, size=(10))
    with open("./tests/random_seeds.txt", 'w') as file:
        for seed in seeds:
            file.write(str(seed))
            file.write('\n')

        

