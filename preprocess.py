# Data Mining - 19KHMT1 - Lab 01: Preprocessing
# Student information
# 19127292 - Nguyen Thanh Tinh
# 19127496 - Truong Quang Minh Nhat

# Import libraries
import argparse  # Agrument
import pandas as pd  # Read and write CSV file


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
    # j = 0
    # while j < len(data[0]):
    #     for i in range(len(data)):
    #         if data[i][j] != data[i][j]:
    #             print(data[0][j])
    #             break
    #     j += 1
    for j in range(len(data[0])):
        for i in range(1, len(data)):
            if data[i][j] != data[i][j]:
                print(data[0][j])
                break


# 2. Count the number of lines with missing data
def count_missing(data):
    ans = 0

    for ai in data:
        for aij in ai:
            if aij != aij:
                ans = ans + 1
                break

    print(ans)


# 3. Fill in the missing value
# Calculate mean of an attribute
def mean(data, col):
    count = 0
    s = 0
    for i in range(1, len(data)):
        if data[i][col] != data[i][col]:
            continue
        count += 1
        s += data[i][col]

    return s / count


# Calculate median of an attribute
def median(data, col):
    val_list = []
    for i in range(1, len(data)):
        if data[i][col] != data[i][col]:
            continue
        val_list.append(data[i][col])
    n = len(val_list)
    val_list.sort()
    if n % 2 == 0:
        median1 = val_list[n // 2]
        median2 = val_list[n // 2 - 1]
        ans = (median1 + median2) / 2
    else:
        ans = val_list[n // 2]

    return ans


# Calculate mode of an attribute
def mode(data, col):
    val_list = []
    for i in range(1, len(data)):
        if data[i][col] in val_list or data[i][col] != data[i][col]:
            continue
        else:
            val_list.append(data[i][col])
    count_list = list(0 for _ in range(0, len(val_list)))
    for value in val_list:
        count_list[val_list.index(value)] += 1

    return val_list[count_list.index(max(count_list))]


# Check Value whether it is an Int,Float or Categorical
def check_value(data, col):
    for i in range(1, len(data)):
        if data[i][col] == data[i][col]:
            if isinstance(data[i][col], int):
                return 1  # int
            elif isinstance(data[i][col], float):
                return 0  # float
            else:
                return -1  # categorical


def fill_missing(data, method, column, output_path):
    if method == 'mean':
        if check_value(data, column) >= 0:
            if check_value(data, column) == 1:
                mean_value = round(mean(data, column))
            else:
                mean_value = mean(data, column)

            for i in range(1, len(data)):
                if data[i][column] != data[i][column]:
                    data[i][column] = mean_value
            write_data_to_file(output_path, data)
        else:
            print("Wrong method for this type of value")

    elif method == 'median':
        if check_value(data, column) >= 0:
            if check_value(data, column) == 1:
                median_value = round(median(data, column))
            else:
                median_value = median(data, column)

            for i in range(1, len(data)):
                if data[i][column] != data[i][column]:
                    data[i][column] = median_value
            write_data_to_file(output_path, data)
        else:
            print("Wrong method for this type of value")

    elif method == 'mode':
        if check_value(data, column) == -1:
            mode_value = mode(data, column)
            for i in range(1, len(data)):
                if data[i][column] != data[i][column]:
                    data[i][column] = mode_value
            write_data_to_file(output_path, data)
        else:
            print("Wrong method for this type of value")


# 4. Remove missing rows with a given missing scale threshold
def missing_rate_row(arr):
    ans = 0

    for item in arr:
        if pd.isna(item):
            ans = ans + 1

    return ans / len(arr)


def remove_row_missing(data, threshold, output_path):
    data_ans = [ai for ai in data if missing_rate_row(ai) < threshold]
    write_data_to_file(output_path, data_ans)


# 5. Remove missing columns with a given missing scale threshold
# Find missing rate of column
def missing_rate_col(data, col):
    count = 0
    for i in range(1, len(data)):
        if data[i][col] != data[i][col]:
            count += 1

    return round((count / len(data)), 2)


def remove_column_missing(data, threshold, output_path):
    for j in range(len(data[0]) - 1, 0, -1):
        if missing_rate_col(data, j) > threshold:
            for ai in data:
                del ai[j]

    write_data_to_file(output_path, data)


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
# find min, max of attribute
def find_min_max(data, col):
    val_list = []
    for i in range(1, len(data)):
        if data[i][col] == data[i][col]:
            val_list.append(data[i][col])
    return min(val_list), max(val_list)


# min-max normalization
def min_max_normalization(data, column, new_min, new_max, output_path):
    if check_value(data, column) >= 0:
        min_val, max_val = find_min_max(data, column)
        for i in range(1, len(data)):
            if data[i][column] == data[i][column]:
                data[i][column] = ((data[i][column] - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min
        write_data_to_file(output_path, data)
    else:
        print("This is categorical attribute")


# count non-Nan value of attribute
def count_value(data, col):
    count = 0
    for i in range(1, len(data)):
        if data[i][col] != data[i][col]:
            continue
        count += 1

    return count


# find StdDev of attribute
def find_standard_deviation(data, col):
    sum_value = 0
    for i in range(1, len(data)):
        if data[i][col] == data[i][col]:
            sum_value += (data[i][col] - mean(data, col)) ** 2
    return (sum_value / count_value(data, col)) ** (1 / 2)


def z_score_normalization(data, column, output_path):
    if check_value(data, column) >= 0:
        mean_value = mean(data, column)
        std_dev = find_standard_deviation(data, column)
        for i in range(1, len(data)):
            if data[i][column] == data[i][column]:
                data[i][column] = (data[i][column] - mean_value) / std_dev
        write_data_to_file(output_path, data)
    else:
        print("This is categorical attribute")


# 8. Calculate attribute expression value
# Create a list of value of an attriute
def list_value_of_an_attribute(data, col):
    val_list = []
    for i in range(1, len(data)):
        val_list.append(data[i][col])
    return val_list


# Convert string in prefix into appropriate data type
def modify_prefix(data, prefix):
    for i in range(len(prefix)):
        if prefix[i] == '+' or prefix[i] == '-' or prefix[i] == '*' or prefix[i] == '/':
            continue
        if prefix[i] in data[0]:
            index = data[0].index(prefix[i])
            prefix[i] = list_value_of_an_attribute(data, index)  # convert attribute's name to attribute's list of value
        elif prefix[i].isdecimal():
            prefix[i] = int(prefix[i])  # if string is an integer
        else:
            prefix[i] = float(prefix[i])  # if string is a float

    return prefix


# split input expression
def split_expression(expression):
    expression_list = []
    operand = ''
    for char in expression:
        if char == ' ':
            continue
        elif char != '+' and char != '-' and char != '*' and char != '/':
            operand += char
        else:
            expression_list.extend([operand, char])
            operand = ''
    expression_list.append(operand)

    return expression_list


# get the priority of operator
def get_priority(operator):
    if operator == '-' or operator == '+':
        return 0
    if operator == '*' or operator == '/':
        return 1


# convert infix expression to prefix expression
def convert_infix_to_prefix(infix):
    infix.reverse()
    prefix = []
    operator_stack = []

    for element in infix:
        if element != '+' and element != '-' and element != '*' and element != '/':
            prefix.append(element)
        else:
            if not operator_stack:
                operator_stack.append(element)
            elif get_priority(element) >= get_priority(operator_stack[-1]):
                operator_stack.append(element)
            elif get_priority(element) < get_priority(operator_stack[-1]):
                while operator_stack and get_priority(operator_stack[-1]) > get_priority(element):
                    # prefix.append(operator_stack[-1])
                    # operator_stack.pop()
                    prefix.append(operator_stack.pop())
                operator_stack.append(element)

    operator_stack.reverse()
    prefix.extend(operator_stack)

    prefix.reverse()
    return prefix


# plus two operand
def plus(op1, op2):
    res = []
    if (not isinstance(op1, list)) or (not isinstance(op2, list)):  # if either op1 or op2 is a number not a list
        if isinstance(op1, list):
            for value in op1:
                res.append(value + op2)
        else:
            for value in op2:
                res.append(op1 + value)
    else:  # both op1 and op2 are lists
        zip_op = zip(op1, op2)
        for op1_i, op2_i in zip_op:
            res.append(op1_i + op2_i)

    return res


# minus two operand
def minus(op1, op2):
    res = []
    if (not isinstance(op1, list)) or (not isinstance(op2, list)):  # if either op1 or op2 is a number not a list
        if isinstance(op1, list):
            for value in op1:
                res.append(value - op2)
        else:
            for value in op2:
                res.append(op1 - value)
    else:  # both op1 and op2 are lists
        zip_op = zip(op1, op2)
        for op1_i, op2_i in zip_op:
            res.append(op1_i - op2_i)

    return res


# multiply two operand
def multiply(op1, op2):
    res = []
    if (not isinstance(op1, list)) or (not isinstance(op2, list)):  # if either op1 or op2 is a number not a list
        if isinstance(op1, list):
            for value in op1:
                res.append(value * op2)
        else:
            for value in op2:
                res.append(op1 * value)
    else:  # both op1 and op2 are lists
        zip_op = zip(op1, op2)
        for op1_i, op2_i in zip_op:
            res.append(op1_i * op2_i)

    return res


# divide two operand
def divide(op1, op2):
    res = []
    if (not isinstance(op1, list)) or (not isinstance(op2, list)):  # if either op1 or op2 is a number not a list
        if isinstance(op1, list):
            for value in op1:
                res.append(value / op2)
        else:
            for value in op2:
                res.append(op1 / value)
    else:  # both op1 and op2 are lists
        zip_op = zip(op1, op2)
        for op1_i, op2_i in zip_op:
            res.append(op1_i / op2_i)

    return res


# calculate the prefix expression
def evaluate_prefix(prefix):
    stack = []

    for value in prefix[::-1]:
        if value != '+' and value != '-' and value != '*' and value != '/':
            stack.append(value)
        else:
            op1 = stack.pop()
            op2 = stack.pop()

            if value == '+':
                stack.append(plus(op1, op2))
            elif value == '-':
                stack.append(minus(op1, op2))
            if value == '*':
                stack.append(multiply(op1, op2))
            if value == '/':
                stack.append(divide(op1, op2))

    return stack.pop()


def calculate(data, input_expression, output_path):
    infix = split_expression(input_expression)
    prefix = convert_infix_to_prefix(infix)

    prefix = modify_prefix(data, prefix)
    res = evaluate_prefix(prefix)

    data[0].append(input_expression)

    for i in range(1, len(data)):
        data[i].append(res[i - 1])

    write_data_to_file(output_path, data)


# Main function
def main():
    # Agruments processing
    parser = argparse.ArgumentParser(description='Preprocessing a CSV data file.')

    parser.add_argument(
        '--task', required=True,
        choices=['ListMissing', 'CountMissing', 'FillMissing', 'RemoveRowMissing', 'RemoveColumnMissing',
                 'RemoveDuplicate', 'MinMaxNormalization', 'Z_ScoreNormalization', 'Calculate'],
        help='Choose a task to do.')
    parser.add_argument('--input_path', required=True, help="input_path CSV file path.")
    parser.add_argument('--output_path', help="output_path CSV file path.")
    parser.add_argument('--method', choices=['mean', 'median', 'mode'], help='Choose a method to fill.')
    parser.add_argument('--threshold', type=float, help='Threshold of Removing tasks.')
    parser.add_argument('--column', type=int, help='Choose a column to fill.')
    parser.add_argument('--new_min', type=float, help='Choose new min for normalization.')
    parser.add_argument('--new_max', type=float, help='Choose new max for normalization.')
    parser.add_argument('--inputExpression', type=str, help='Input an expression')

    args = parser.parse_args()

    # Read data from file
    data = read_data_from_file(args.input_path)

    # Base on args, do the corresponding task
    if args.task == "ListMissing":
        list_missing(data)
    elif args.task == 'CountMissing':
        count_missing(data)
    elif args.task == 'FillMissing':
        fill_missing(data, args.method, args.column, args.output_path)
    elif args.task == 'RemoveRowMissing':
        remove_row_missing(data, args.threshold, args.output_path)
    elif args.task == 'RemoveColumnMissing':
        remove_column_missing(data, args.threshold, args.output_path)
    elif args.task == 'RemoveDuplicate':
        remove_duplicate(data, args.output_path)
    elif args.task == 'MinMaxNormalization':
        min_max_normalization(data, args.column, args.new_min, args.new_max, args.output_path)
    elif args.task == 'Z_ScoreNormalization':
        z_score_normalization(data, args.column, args.output_path)
    elif args.task == 'Calculate':
        calculate(data, args.inputExpression, args.output_path)


# Entry point
if __name__ == '__main__':
    main()
