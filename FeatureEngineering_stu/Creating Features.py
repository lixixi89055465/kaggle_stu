import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
accidents = pd.read_csv("./input/fe-course-data/accidents.csv")
autos = pd.read_csv("./input/fe-course-data/autos.csv")
concrete = pd.read_csv("./input/fe-course-data/concrete.csv")
customer = pd.read_csv("./input/fe-course-data/customer.csv")