# Data Mining - 19KHMT1 - Lab 01: Preprocessing
# Student information
# 19127292 - Nguyen Thanh Tinh
# 19127496 - Truong Quang Minh Nhat

# Import libraries
import argparse      # Command line arguments processing
import pandas as pd  # Read and write CSV file


# 0. Utility functions
# 0.1. Read the data from a CSV file
# input_path: string
# return: list
def read_data_from_file(input_path):
    # Read input data
    df = pd.read_csv(input_path)

    # Convert the dataframe into a list, the first row is attibute names, the other rows are core data
    data = [df.columns.tolist()] + df.values.tolist()
    return data


# 0.2. Write the data to a CSV file
# output_path: string
# data: list
def write_data_to_file(output_path, data):
    # Convert the list into a dataframe, the first row is attibute names, the other rows are core data
    df = pd.DataFrame(data[1:], columns=data[0])

    # Write dataframe to CSV file
    df.to_csv(output_path, index=False)


# 1. List columns with missing data
# data: list
def list_missing(data):
    for j in range(len(data[0])):         # Columns traversal
        for i in range(1, len(data)):     # Check all elements in that columns
            if data[i][j] != data[i][j]:  # If it's different from itself, it's NaN data (missing value)
                print(data[0][j])         # Print column name
                break                     # Go to the next column


# 2. Count the number of lines with missing data
# data: list
def count_missing(data):
    ans = 0                    # Number of lines with missing data

    for ai in data:            # Rows traversal
        for aij in ai:         # Check all elements in that row
            if aij != aij:     # If it's different from itself, it's NaN data (missing value)
                ans = ans + 1  # Increase counter
                break          # Go to the next row

    print(ans)                 # Print answer


# 3. Fill in the missing values
# 3.Util_1. Calculate the mean value of an attribute
# data: list
# col: int - index of attribute to calculate
def mean(data, col):
    count = 0                             # Number of Non NaN values
    sum_value = 0                         # Sum of Non NaN values

    for i in range(1, len(data)):         # Check all elements in column col
        if data[i][col] != data[i][col]:  # If it's different from itself, it's NaN data (missing value)
            continue                      # Go to the next element
        count += 1                        # Else, increase counter
        sum_value += data[i][col]         # And add it to sum value

    return sum_value / count              # Return the mean value


# 3.Util_2. Calculate the median value of an attribute
# data: list
# col: int - index of attribute to calculate
def median(data, col):
    value_list = []                             # List of Non NaN values

    for i in range(1, len(data)):               # Check all elements in column col
        if data[i][col] != data[i][col]:        # If it's different from itself, it's NaN data (missing value)
            continue                            # Go to the next element
        value_list.append(data[i][col])         # Else, append it to value list

    value_list.sort()                           # Sort the value list

    n = len(value_list)                         # Get the number of values in value list
    if n % 2 == 0:                              # If the number of values is even
        median_left = value_list[n // 2]        #
        median_right = value_list[n // 2 - 1]   # The answer is average of two middle elements
        ans = (median_left + median_right) / 2  #
    else:                                       # Else the number of values is odd
        ans = value_list[n // 2]                # The answer is the middle element

    return ans                                  # Return the median value


# 3.Util_3. Calculate the mode value of an attribute
# data: list
# col: int - index of attribute to calculate
def mode(data, col):
    # Create value list
    value_list = []                                                     # List of values
    for i in range(1, len(data)):                                       # Check all elements in column col
        if data[i][col] in value_list or data[i][col] != data[i][col]:  # If existed in value list, or NaN value
            continue                                                    # Go to the next element
        else:                                                           # Else
            value_list.append(data[i][col])                             # Append it into value list

    # Create counting list corresponding to value list
    count_list = [0 for _ in range(len(value_list))]                    # List of counters corresponding to value list
    for i in range(1, len(data)):                                       # Check all elements in column col
        if data[i][col] == data[i][col]:                                # If it's not an NaN value
            count_list[value_list.index(data[i][col])] += 1             # Increase it's counter

    # Return the value which one has maximum counter
    return value_list[count_list.index(max(count_list))]


# 3.Util_4. Get type of column whether it is Int, Float or Categorical
# data: list
# col: int - index of attribute to calculate
# return: 1 - Int, 0 - Float, -1 - Categorical
def get_type(data, col):
    for i in range(1, len(data)):          # Check all elements in column col
        if data[i][col] == data[i][col]:   # Find the first element that is not a NaN value
            if data[i][col].is_integer():  # If it's an Int value
                return 1                   # It's an Int column
            elif data[i][col].is_float():  # Else if it's an Float value
                return 0                   # It's a Float column
            else:                          # Else
                return -1                  # It's a Categorical column


# 3. Fill in the missing values
# data: list
# method: string - the method to process (mean, median or mode)
# col: int - column index
# output_path: string - the output path
def fill_missing(data, method, col, output_path):
    # Mean method
    if method == 'mean':
        if get_type(data, col) >= 0:                       # Mean method only works on Numeric data
            if get_type(data, col) == 1:                   # If it's Int column
                mean_value = round(mean(data, col))        # Mean value should be rounded
            else:                                          # Else
                mean_value = mean(data, col)               # Mean value doesn't need to be rounded

            for i in range(1, len(data)):                  # Traversal all elements in column col
                if data[i][col] != data[i][col]:           # If it's a NaN value (missing value)
                    data[i][col] = mean_value              # Fill it

            write_data_to_file(output_path, data)          # Save changes
        else:
            print('Wrong method for this type of value!')  # Mean method doesn't work on Categorical data

    # Median Method
    elif method == 'median':
        if get_type(data, col) >= 0:                       # Median method only works on Numeric data
            if get_type(data, col) == 1:                   # If it's Int column
                median_value = round(median(data, col))    # Median value should be rounded
            else:                                          # Else
                median_value = median(data, col)           # Median value doesn't need to be rounded

            for i in range(1, len(data)):                  # Traversal all elements in column col
                if data[i][col] != data[i][col]:           # If it's a NaN value (missing value)
                    data[i][col] = median_value            # Fill it

            write_data_to_file(output_path, data)          # Save changes
        else:
            print('Wrong method for this type of value!')  # Median method doesn't work on Categorical data

    # Mode method
    elif method == 'mode':
        if get_type(data, col) == -1:                      # Mode method only works on Categorical data
            mode_value = mode(data, col)                   # Find mode value

            for i in range(1, len(data)):                  # Traversal all elements in column col
                if data[i][col] != data[i][col]:           # If it's a NaN value (missing value)
                    data[i][col] = mode_value              # Fill it

            write_data_to_file(output_path, data)          # Save changes
        else:
            print('Wrong method for this type of value')   # Mode method doesn't work on Numeric data


# 4. Remove missing rows with a given missing rate threshold
# 4.Util. Calculate missing rate of a row
# row: list - the row to be calculated
# return: float - (0; 1) - missing rate
def missing_rate_row(row):
    count = 0                  # Counting missing values in a row

    for item in row:           # Check all elements in the row
        if item != item:       # If it's different from itself, it's NaN data (missing value)
            count = count + 1  # Increase counter

    return count / len(row)    # Return missing rate


# 4. Remove missing rows with a given missing rate threshold
# data: list
# threshold: float - (0; 1) - missing rate threshold
# output_path: string
def remove_row_missing(data, threshold, output_path):
    # Choose rows which have missing rate smaller than threshold
    data_ans = [ai for ai in data if missing_rate_row(ai) < threshold]

    # Save changes
    write_data_to_file(output_path, data_ans)


# 5. Remove missing columns with a given missing rate threshold
# 5.Util. Calculate missing rate of a column
# data: list
# column: int - index of the column to be calculated
# return: float - (0; 1) - missing rate
def missing_rate_col(data, col):
    count = 0                             # Counting missing values in a column

    for i in range(1, len(data)):         # Check all elements in the column col
        if data[i][col] != data[i][col]:  # If it's different from itself, it's NaN data (missing value)
            count += 1                    # Increase counter

    return count / (len(data) - 1)        # Return missing rate


# 5. Remove missing columns with a given missing rate threshold
# data: list
# threshold: float - (0; 1) - missing rate threshold
# output_path: string
def remove_column_missing(data, threshold, output_path):
    for j in range(len(data[0]) - 1, 0, -1):       # Delete from right to left, so the code will be cleaner
        if missing_rate_col(data, j) > threshold:  # If missing rate of the column is greater than threshold
            for ai in data:                        # With each row in data
                del ai[j]                          # Delete the element number j-th

    write_data_to_file(output_path, data)          # Save changes


# 6. Remove duplicate instances
# data: list
# output_pathL string
def remove_duplicate(data, output_path):
    data_ans = [data[0], data[1]]                               # The first row and the second row are kept

    for i in range(2, len(data)):                               # Traversal all rows
        is_unique = True                                        # Supposing that the current row is unique

        row_i = [item for item in data[i] if item == item]      # Copy Non NaN item in the current row
        row_i = row_i[1:]                                       # Exclude the first item, it's ID attribute

        # Compare the current row to the before rows
        for j in range(1, i):                                   # j is before row index
            row_j = [item for item in data[j] if item == item]  # Copy Non NaN item in the j-th row
            row_j = row_j[1:]                                   # Exclude the first item, it's ID attribute

            if row_i == row_j:                                  # If row i-th and row j-th are the same
                is_unique = False                               # The current is not unique
                break                                           # No need to compare with other rows

        if is_unique:                                           # If the current row is unique
            data_ans = data_ans + [data[i]]                     # Add it the answe data

    write_data_to_file(output_path, data_ans)                   # Save changes


# 7. Normalize a numeric attribute
# 7.a. Min-max method
# 7.a.Util. Find min value and max value of an attribute
# data: list
# col: int - index of column to be calculated
# return: int, int - the minimum value and the maximum value
def find_min_max(data, col):
    value_list = []                          # List of Non NaN values in column col

    for i in range(1, len(data)):            # Check all elements in column col
        if data[i][col] == data[i][col]:     # If it's a Non NaN value
            value_list.append(data[i][col])  # Append it to value list

    return min(value_list), max(value_list)  # Return the minimum value and the maximum value


# 7.a. Min-max normalization
# data: list
# col: int - index of column to be normalized
# new_min, new_max: int, int
# output_path: string
def min_max_normalization(data, col, new_min, new_max, output_path):
    # Check for Numeric type
    if get_type(data, col) >= 0:
        # Find old minimum value and old maximum value
        old_min, old_max = find_min_max(data, col)

        # Update on all element on column col
        for i in range(1, len(data)):
            if data[i][col] == data[i][col]:  # Only update Non NaN elements
                data[i][col] = ((data[i][col] - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

        # Save changes
        write_data_to_file(output_path, data)

    # Min-max normalization method doesn't work on Categorical data
    else:
        print('Cannot process on categorical attribute!')


# 7.b.Util_1. Count Non NaN values of an attribute
# data: list
# col: int - index of the column to be calculated
# return: int - number of Non NaN values
def count_value(data, col):
    count = 0                             # Number of Non NaN values in column col

    for i in range(1, len(data)):         # Check all elements in column col
        if data[i][col] == data[i][col]:  # If it's a Non NaN value
            count += 1                    # Increase counter

    return count                          # Return number of Non NaN values


# 7.b.Util_2. Find StdDev of an attribute
# data: list
# col: int - index of the column to be calculated
# return: float - StdDev of that column
def find_standard_deviation(data, col):
    sum_value = 0                                               # Sum value of column col
    mean_value = mean(data, col)                                # Calculate the mean value of column col

    for i in range(1, len(data)):                               # Check all elements in column col
        if data[i][col] == data[i][col]:                        # If it's a Non NaN value
            sum_value += (data[i][col] - mean_value) ** 2       # Add variance to sum value

    return (sum_value / count_value(data, col)) ** (1 / 2)      # Return StdDev


# 7.b. Z-score normalization
# data: list
# col: int - index of column to be normalized
# output_path: string
def z_score_normalization(data, col, output_path):
    # Check for Numeric type
    if get_type(data, col) >= 0:
        # Calculate importants values: mean and StdDev
        mean_value = mean(data, col)
        std_dev = find_standard_deviation(data, col)

        for i in range(1, len(data)):                                 # Check all elements in column col
            if data[i][col] == data[i][col]:                          # If it's a Non NaN value
                data[i][col] = (data[i][col] - mean_value) / std_dev  # Update it

        write_data_to_file(output_path, data)                         # Save changes

    # Z-score normalization method doesn't work on Categorical data
    else:
        print('Cannot process on categorical attribute!')


# 8. Calculate attribute expression value
# 8.Util_1. Create a list of value of an attriute
# data: list
# col: int - index of column to be get values
def get_value_list(data, col):
    value_list = []                      # List of values in column col

    for i in range(1, len(data)):      # Check all elements in column col
        value_list.append(data[i][col])  # Append it to the list

    return value_list                    # Return the value list


# 8.Util_2. Convert a string in prefix into appropriate data type
# data: list
# prefix: string
def modify_prefix(data, prefix):
    # Traversal the prefix string
    for i in range(len(prefix)):
        # If it's operator, skip it
        if prefix[i] == '+' or prefix[i] == '-' or prefix[i] == '*' or prefix[i] == '/':
            continue

        # If it's an operand, convert prefix[i] to correspoding value list
        if prefix[i] in data[0]:
            index = data[0].index(prefix[i])
            prefix[i] = get_value_list(data, index)

        # If it's a decimal value, parse it into integer
        elif prefix[i].isdecimal():
            prefix[i] = int(prefix[i])

        # Else, it's a float value, parse it into float
        else:
            prefix[i] = float(prefix[i])

    return prefix


# 8.Util_3. Split input expression
# expression: string
def split_expression(expression):
    expression_list = []                                       # List of all elements in expression
    operand = ''                                               # A string to store operands

    for c in expression:                                       # Check all characters in expression
        if c == ' ':                                           # If it's a blank
            continue                                           # Skip it
        elif c != '+' and c != '-' and c != '*' and c != '/':  # Else if it's an operand
            operand += c                                       # Temporary add it into operand storage
        else:                                                  # Else if it's an operator
            expression_list.extend([operand, c])               # Update operand and operator
            operand = ''                                       # Refresh the operand storage

    expression_list.append(operand)                            # Append the last operand

    return expression_list                                     # Return expression list


# 8.Util_4. Get the priority of operator
# operator: char
# return: int [0, 1] - the priority of the operator, 1 is higher than 0
def get_priority(operator):
    if operator == '+' or operator == '-':  # If it's '+' or '-'
        return 0                            # Low priority
    if operator == '*' or operator == '/':  # If it's '*' or '/'
        return 1                            # High priority


# 8.Util_5. Convert infix expression to prefix expression
# infix: string
# return: string - prefix expression
def convert_infix_to_prefix(infix):
    # Reverse the infix expression
    infix.reverse()

    # Preparing for prefix expression
    prefix = []
    operator_stack = []

    # Check all elements in infix expression
    for element in infix:
        # If it's an operand, append it to prefix
        if element != '+' and element != '-' and element != '*' and element != '/':
            prefix.append(element)

        # Else if it's an operator
        else:
            # Push the operator if the stack is empty, or high priority
            # If it's a low priority, process the high priority operators in the stack first
            if not operator_stack:
                operator_stack.append(element)
            elif get_priority(element) >= get_priority(operator_stack[-1]):
                operator_stack.append(element)
            elif get_priority(element) < get_priority(operator_stack[-1]):
                while operator_stack and get_priority(operator_stack[-1]) > get_priority(element):
                    prefix.append(operator_stack.pop())
                operator_stack.append(element)

    # Combine prefix expression and operator stack
    operator_stack.reverse()
    prefix.extend(operator_stack)

    # Reverse prefix expression and return the answer
    prefix.reverse()
    return prefix


# 8.Util_6. Plus two operand
# op1, op2: list
def plus(op1, op2):
    ans = []

    if (not isinstance(op1, list)) or (not isinstance(op2, list)):  # If either op1 or op2 is a number not a list
        # Do "Broadcast" calculate
        if isinstance(op1, list):
            for value in op1:
                ans.append(value + op2)
        else:
            for value in op2:
                ans.append(op1 + value)
    else:  # Else both op1 and op2 are lists
        zip_op = zip(op1, op2)
        for op1_i, op2_i in zip_op:
            ans.append(op1_i + op2_i)

    return ans


# 8.Util_7. Minus two operand
# op1, op2: list
def minus(op1, op2):
    ans = []

    if (not isinstance(op1, list)) or (not isinstance(op2, list)):  # If either op1 or op2 is a number not a list
        # Do "Broadcast" calculate
        if isinstance(op1, list):
            for value in op1:
                ans.append(value - op2)
        else:
            for value in op2:
                ans.append(op1 - value)
    else:  # Else both op1 and op2 are lists
        zip_op = zip(op1, op2)
        for op1_i, op2_i in zip_op:
            ans.append(op1_i - op2_i)

    return ans


# 8.Util_8. Multiply two operand
# op1, op2: list
def multiply(op1, op2):
    ans = []

    if (not isinstance(op1, list)) or (not isinstance(op2, list)):  # If either op1 or op2 is a number not a list
        # Do "Broadcast" calculate
        if isinstance(op1, list):
            for value in op1:
                ans.append(value * op2)
        else:
            for value in op2:
                ans.append(op1 * value)
    else:  # Else both op1 and op2 are lists
        zip_op = zip(op1, op2)
        for op1_i, op2_i in zip_op:
            ans.append(op1_i * op2_i)

    return ans


# 8.Util_9. Devide two operand
# op1, op2: list
def divide(op1, op2):
    ans = []

    if (not isinstance(op1, list)) or (not isinstance(op2, list)):  # If either op1 or op2 is a number not a list
        # Do "Broadcast" calculate
        if isinstance(op1, list):
            for value in op1:
                if op2 == 0:
                    ans.append('')
                else:
                    ans.append(value // op2)
        else:
            for value in op2:
                if value == 0:
                    ans.append('')
                else:
                    ans.append(op1 // value)
    else:  # Else both op1 and op2 are lists
        zip_op = zip(op1, op2)
        for op1_i, op2_i in zip_op:
            if op2_i == 0:
                ans.append('')
            else:
                ans.append(op1_i // op2_i)

    return ans


# 8.Util_10. Calculate the prefix expression
# prefix: list
def calculate_prefix(prefix):
    # Prepare a stack of operands
    stack = []

    # Traversal all item in prefix expression in reverse order
    for value in prefix[::-1]:
        # If it's an operand, push it into th stack
        if value != '+' and value != '-' and value != '*' and value != '/':
            stack.append(value)

        # Else if it's an operator, pop the two latest operands and calculate them
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

    # Return the last one item in the stack, it is the answer
    return stack.pop()


# 8. Calculate attributes expression
# data: list
# input_expression: string
# output_expression: string
def calculate(data, input_expression, output_path):
    # Get infix expression
    infix = split_expression(input_expression)

    # Get prefix expression
    prefix = convert_infix_to_prefix(infix)
    prefix = modify_prefix(data, prefix)

    # Calculate the prefix expression, save it to a list named ans
    ans = calculate_prefix(prefix)

    # Append new data
    data[0].append(input_expression)

    for i in range(1, len(data)):
        data[i].append(ans[i - 1])

    # Save changes
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
    parser.add_argument('--input_expression', type=str, help='Input an expression')

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
        calculate(data, args.input_expression, args.output_path)


# Entry point
if __name__ == '__main__':
    main()
