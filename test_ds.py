import sys
import pandas as pd


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_ds.py <path_to_parquet>")
        return

    path = sys.argv[1]
    df = pd.read_parquet(path)
    print(df.head())


if __name__ == "__main__":
    main()
