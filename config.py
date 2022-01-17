IS_NOMINAL = True  # Are values in dataset nominal
DATA_FILENAME = '2d-3c-no123.arff'  # Name of file with data
RATIO = 0.5  # Ratio of sample_size / data_size
NUM_CLUSTERS = 3  # Number of output clusters
THETA = 0.0  # ROCK parameter
MAX_DISTANCE = 0.6  # Only for nominal problems, maximum distance between points to be neighbours
IS_LABEL_FIRST = False  # Only for categorical problems, if class label is in first row or in the last

# Predefined paths
NOMINAL_DATA_PATH = 'data/nominal'
CATEGORICAL_DATA_PATH = 'data/categorical'
OUTPUT_PATH = 'results/outputs'
TARGET_PATH = 'results/targets'
CATEGORICAL_PLOTS_PATH = 'results/categorical'

# For plotting
DATA_ATTRIBUTES = ['x', 'y']
DATA_TARGET = 'class'

C_RATIO = 0.1
N_RATIO = 0.5

C_DATA_ATTRIBUTE = 'cluster'

# config  values for testing
N_FILENAMES = [
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

N_CLUSTERS_SIZE = [3, 4, 3, 2, 3, 2, 2, 2, 16, 4, 4]

C_FILENAMES = [
    'agaricus-lepiota.data',
    'breast-cancer.data',
    'adult.data'
]

C_CLUSTERS_SIZE = [20, 2, 2]
LABEL_FIRST = [True, False, False]

THRESHOLDS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DISTANCES = [0.4, 0.8, 1.6, 3.2, 6.4]
