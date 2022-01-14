NOMINAL_DATA_PATH = 'data/nominal'
CATEGORICAL_DATA_PATH = 'data/categorical'
OUTPUT_PATH = 'results/outputs'
TARGET_PATH = 'results/targets'
CATEGORICAL_PLOTS_PATH = 'results/categorical'

# For plotting
DATA_ATTRIBUTES = ['x', 'y']
DATA_TARGET = 'class'

CATEGORICAL_DATA_ATTRIBUTE = 'cluster'
USE_PARALLEL = False

N_TEST_FILENAMES = [
    '2d-3c-no123.arff',
    '2d-4c-no4.arff',
    '3-spiral.arff',
    'banana.arff',
    'blobs.arff',
    'circle.arff',
    'curves1.arff',
    'donut1.arff',
    's-set1.arff',
    'sizes1.arff',
    'triangle1.arff'
]

N_NUM_CLUSTERS = [3, 4, 3, 2, 3, 2, 2, 2, 16, 4, 4]

C_TEST_FILENAMES = [
    'agaricus-lepiota.data',
    'breast-cancer.data',
    'adult.data'
]

C_NUM_CLUSTERS = [20, 2, 2]
LABEL_FIRST = [True, False, False]

THRESHOLDS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DISTANCES = [0.4, 0.8, 1.6, 3.2, 6.4]
