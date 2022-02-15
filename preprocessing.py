# All prerpocessing functions are provided in this file


def pre_processing():
    """
    preprocessing class takes few parameters as input and apply requested
    preprocessing steps on data
    methods are wraper around sklearn package preprocessing module

    data: input data (sample * dimension)
    method: name of preprocessing method needs to be applied
            'normalization'
            'standarization'
            'minmax'
            'Kbins_discretization'
            'polynomial'
                .
                .
                .

    """
    # importing sklearn preprocessing module
    from sklearn import preprocessing

    return preprocessing
