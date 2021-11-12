# Data Mining - 19KHMT1 - Lab 01: Preprocessing
# Student information
# 19127292 - Nguyen Thanh Tinh
# 19127496 - Truong Quang Minh Nhat

# Import libraries
import argparse
import pandas as pd


# 0. Utility functions
# 0.1. Read the data from a CVS file
# input_path: string
# return: list
def read_data_from_file(input_path):
    # Read input data
    df = pd.read_csv(input_path)

    # Convert the dataframe into a list, the first row is attibute names, the other rows are core data
    data = [df.columns.tolist()] + df.values.tolist()
    return data


# 0.2. Write the data
# output_path: string
# data: list
def write_data_to_file(output_path, data):
    # Convert the list into a dataframe, the first row is attibute names, the other rows are core data
    df = pd.DataFrame(data[1:], columns=data[0])

    # Write dataframe to CSV file
    df.to_csv(output_path, index=False)


# 1. List columns with missing data
def list_missing(data):
    print('List Missing')


# 2. Count the number of lines with missing data
def count_missing(data):
    ans = 0

    for ai in data:
        for aij in ai:
            if pd.isna(aij):
                ans = ans + 1
                break

    print(ans)


# 3. Fill in the missing value
def fill_missing(data, method, output_path):
    if method == 'mean':
        pass
    elif method == 'median':
        pass
    elif method == 'mode':
        pass

    write_data_to_file(output_path, data)


# 4. Remove missing rows with a given missing scale threshold
def missing_rate(arr):
    ans = 0

    for item in arr:
        if pd.isna(item):
            ans = ans + 1

    return ans / len(arr)


def remove_row_missing(data, threshold, output_path):
    data_ans = [ai for ai in data if missing_rate(ai) < threshold]
    write_data_to_file(output_path, data_ans)


# 5. Remove missing columns with a given missing scale threshold
def remove_column_missing(data):
    print('Remove Column Missing')


# 6. Remove duplicate instances
def remove_duplicate(data, output_path):
    data_ans = [data[0], data[1]]

    for i in range(2, len(data)):
        is_unique = True
        for j in range(1, i):
            row_i = [item for item in data[i] if not pd.isna(item)]
            row_i = row_i[1:]

            row_j = [item for item in data[j] if not pd.isna(item)]
            row_j = row_j[1:]

            if row_i == row_j:
                is_unique = False
                break

        if is_unique:
            data_ans = data_ans + [data[i]]

    write_data_to_file(output_path, data_ans)


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
    parser.add_argument('--input_path', required=True, help="input_path CSV file path.")
    parser.add_argument('--output_path', help="output_path CSV file path.")
    parser.add_argument('--method', choices=['mean', 'median', 'mode'], help='Choose a method to fill.')
    parser.add_argument('--threshold', type=float, help='Threshold of Removing tasks.')

    args = parser.parse_args()

    # Read data from file
    data = read_data_from_file(args.input_path)

    # Base on args, do the corresponding task
    if args.task == "ListMissing":
        list_missing(data)
    elif args.task == 'CountMissing':
        count_missing(data)
    elif args.task == 'FillMissing':
        fill_missing(data, args.method, args.output_path)
    elif args.task == 'RemoveRowMissing':
        remove_row_missing(data, args.threshold, args.output_path)
    elif args.task == 'RemoveColumnMissing':
        remove_column_missing(data)
    elif args.task == 'RemoveDuplicate':
        remove_duplicate(data, args.output_path)
    elif args.task == 'Normalize':
        normalize(data)
    elif args.task == 'Calculate':
        calculate(data)


# Entry point
if __name__ == '__main__':
    main()