"""
Plotting columns of a CSV on a single plot. Temporary util script.
"""

import argparse
import csv
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


def plot(csv_path: str) -> None:
    """
    Plot the values in the CSV file at `csv_path`. We treat each column of the CSV as a
    curve to plot.
    """

    # Read in data.
    cols = []
    with open(csv_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row_num, row in enumerate(reader):
            if row_num == 0:
                cols = [[] for _ in range(len(row))]
            for i, element in enumerate(row):
                cols[i].append(float(element))

    if len(cols) == 0:
        print("Empty CSV file!")
        exit()
    x = list(range(len(cols[0])))

    # Plot each column.
    for i, col in enumerate(cols):
        plt.plot(x, col, label=str(i))

    # Show plot.
    pos = csv_path.rfind(".")
    plot_path = csv_path[:pos] + "_plot.png"
    plt.legend()
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Plotting the columns of a CSV on a single plot."
    )
    parser.add_argument("csv_path", type=str, help="Path of CSV file to plot.")
    args = parser.parse_args()

    plot(args.csv_path)
