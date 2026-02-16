"""
Data processing
"""

from sklearn.model_selection import train_test_split
from neuromancer.dataset import DictDataset


def data_split(datadict, test_size=0, val_size=0, random_state=42):
    """
    Split data into training, validation, and test sets based on provided sizes
    """
    # init empty training set
    datasets = {"train":None}
    # create index range based on the input dataset size
    ind_train = range(len(next(iter(datadict.values()))))
    # splitting for test dataset if test_size is provided
    if test_size:
        # data split
        ind_train, ind_test = train_test_split(ind_train,
                                               test_size=test_size,
                                               random_state=random_state,
                                               shuffle=True)
        # test dataset
        datasets["test"] = DictDataset({key: param[ind_test] for key, param in datadict.items()}, name="test")
    # splitting for validation dataset if val_size is provided
    if val_size:
        # data split
        ind_train, ind_dev = train_test_split(ind_train,
                                              test_size=val_size,
                                              random_state=random_state,
                                              shuffle=True)
        # validation dataset
        datasets["dev"] = DictDataset({key: param[ind_dev] for key, param in datadict.items()}, name="dev")
    # final assignment of remaining indices to training dataset
    datasets["train"] = DictDataset({key: param[ind_train] for key, param in datadict.items()}, name="train")
    return tuple(datasets.values())
