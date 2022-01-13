DATA_PATH = 'data'
OUTPUT_PATH = 'results/outputs'
TARGET_PATH = 'results/targets'
CATEGORICAL_PATH = 'results/categorical'

DATA_ATTRIBUTES = ['a0', 'a1']
DATA_TARGET = 'class'

CATEGORICAL_DATA_ATTRIBUTE = 'cluster'
USE_PARALLEL = True

NOMINAL_TEST_FILENAMES = [
    '2d-3c-no123.arff'
]

NOMINAL_NUM_CLUSTERS = [3]

CATEGORICAL_TEST_FILENAMES = [
    'agaricus-lepiota.data'
]

CATEGORICAL_NUM_CLUSTERS = [20]

THRESHOLDS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DISTANCES = [0.2, 0.4, 0.6, 0.8, 1.0]
