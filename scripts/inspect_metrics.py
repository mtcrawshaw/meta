import os
import pickle
import glob


def main():

    metrics_filenames = glob.glob(os.path.join("data", "metrics", "*.pkl"))
    for metrics_filename in metrics_filenames:
        with open(metrics_filename, "rb") as metrics_file:
            metrics = pickle.load(metrics_file)
        print(metrics_filename)
        print(metrics)
        print("")


if __name__ == "__main__":
    main()
