"""
Task Main Module
"""

# import necessary modules
import numpy as np

from restricted_boltzmann_machine import RBM


def main():
    """
    Task Main Method
    """

    # construct restricted boltzmann machine
    rbm = RBM(visible_dim=2, hidden_dim=2)

    # create train set
    train_set = np.random.randint(low=0, high=2, size=(2, 2))
    print(f'train_set.shape: {train_set.shape}')
    print(f'train_set: \n{train_set}')

    visible = train_set[0, :]

    hidden = rbm.train(train_set[0, :])

    return 0


if __name__ == '__main__':
    main()
