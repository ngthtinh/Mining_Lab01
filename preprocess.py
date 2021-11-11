# Data Mining - 19KHMT1 - Lab 01: Preprocessing
# Student information
# 19127292 - Nguyen Thanh Tinh
# 19127496 - Truong Quang Minh Nhat

# Import libraries
import argparse
import pandas as pd


# 1. List columns with missing data
def list_missing(data):
    print('List Missing')


# 2. Count the number of lines with missing data
def count_missing(data):
    print('Count Missing')


# 3. Fill in the missing value
def fill_missing(data):
    print('Fill Missing')


# 4. Remove missing rows with a given missing scale threshold
def remove_row_missing(data):
    print('Remove Row Missing')


# 5. Remove missing columns with a given missing scale threshold
def remove_column_missing(data):
    print('Remove Column Missing')


# 6. Remove duplicate instances
def remove_duplicate(data):
    print('Remove Duplicate')


# 7. Normalize a numeric attribute
def normalize(data):
    print('Normalize')


# 8. Calculate attribute expression value
def calculate(data):
    print('Calculate')


# Main function
def main():
    # Agruments processing
    parser = argparse.ArgumentParser(description='Preprocessing a CSV data file.')

    parser.add_argument(
        '--task', required=True,
        choices=['ListMissing', 'CountMissing', 'FillMissing', 'RemoveRowMissing', 'RemoveColumnMissing',
                 'RemoveDuplicate', 'Normalize', 'Calculate'], help='Choose a task to do.')
    parser.add_argument('--input', required=True, help="Input CSV file path.")
    parser.add_argument('--output', help="Output CSV file path.")

    args = parser.parse_args()

    # Read input data
    df = pd.read_csv(args.input)
    data = df.values.tolist()

    # Base on args, do the corresponding task
    if args.task == "ListMissing":
        list_missing(data)
    elif args.task == 'CountMissing':
        count_missing(data)
    elif args.task == 'FillMissing':
        fill_missing(data)
    elif args.task == 'RemoveRowMissing':
        remove_row_missing(data)
    elif args.task == 'RemoveColumnMissing':
        remove_column_missing(data)
    elif args.task == 'RemoveDuplicate':
        remove_duplicate(data)
    elif args.task == 'Normalize':
        normalize(data)
    elif args.task == 'Calculate':
        calculate(data)


# Entry point
if __name__ == '__main__':
    main()
