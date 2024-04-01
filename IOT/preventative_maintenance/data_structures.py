import tensorflow as tf
import numpy as np
import pandas as pd



SHOW_DATAFRAMES=False

def analyze_object(obj):
    print("------------------------Object Analysis Begins--------------------------")
    # Check if it's a Series
    if isinstance(obj, pd.Series):
        print("Object is a pandas Series")
        print("Shape:", obj.shape)
        print("Length of Series:", obj.shape[0])
        print("Type of elements:", type(obj.iloc[0]))
        print("First 3 elements:")
        print(obj.head(3))
    # Check if it's a DataFrame
    elif isinstance(obj, pd.DataFrame):
        print("Object is a pandas DataFrame")
        print("Shape:", obj.shape)
        print("Length of Series:", len(obj))
        print("Types of elements:")
        print(obj.dtypes)
        for column in obj.columns:
            print(f"Type of elements in column '{column}':", type(obj[column].iloc[0]))
        # Check type of elements in each row
        for index, row in obj.iterrows():
            print(f"Type of elements in row '{index}':", type(row.iloc[0]))
        print("First 3 elements:")
        print(obj.head(3))
    # Check if it's a NumPy array
    elif isinstance(obj, np.ndarray):
        print("Object is a NumPy array")
        print("Shape:", obj.shape)
        print("Dimensions in this array: ", obj.ndim)
        print("Length of NumPy Array:", len(obj))
        print("Type of elements:", obj.dtype)
        print("First 3 elements:")
        print(obj[:3])
    # Check if it's a TensorFlow tensor
    elif tf.is_tensor(obj):
        print("Object is a TensorFlow tensor")
        print("Shape:", obj.shape)
        print("Dimensions in this tensor: ", obj.ndim)
        print("Length of TensorFlow Object:", obj.shape[0])
        print("Data type:", obj.dtype)
        print("First 3 elements:")
        print(obj[:3])
    else:
        print("Unknown type")

    print("------------------------Object Analysis Ends--------------------------")
"""
Data Representation:
DataFrame: A DataFrame is a two-dimensional labeled data structure with columns of potentially different types. It is part of the pandas library and is designed for handling tabular data. a DataFrame is specifically designed to be a two-dimensional labeled data structure. It cannot have more than two dimensions. Each column in a DataFrame represents a different variable, and each row represents a different observation.
NumPy Array: A NumPy array is a grid of values of the same type. It is part of the NumPy library and is used for numerical computing. While arrays can have multiple dimensions, they lack built-in support for row and column labels.

Data Types:
DataFrame: DataFrames can contain columns of different data types (e.g., integers, floats, strings, etc.). Each column in a DataFrame is essentially a pandas Series, which can have its own data type.
NumPy Array: NumPy arrays are homogeneous, meaning all elements in an array must have the same data type. This allows for more efficient storage and operations on numerical data.

Indexing and Labeling:
DataFrame: DataFrames support both integer-based and label-based indexing. Columns and rows can have user-defined labels, making it easy to reference specific subsets of data using these labels. the row and column labels in a DataFrame can be accessed through the index and columns attributes, respectively. In pandas DataFrames, when using .loc[], you first specify the row label and then the column label. 
NumPy Array: NumPy arrays support only integer-based indexing. While you can create separate arrays to store row and column labels, NumPy arrays themselves do not have built-in support for labels.

Functionality:
DataFrame: DataFrames offer a wide range of functionality for data manipulation and analysis, including data alignment, grouping, merging, reshaping, and more. They also support missing data handling and time series functionality.
NumPy Array: NumPy arrays provide fundamental array operations and mathematical functions for numerical computing. They are optimized for numerical operations and are widely used in scientific computing and machine learning.

Libraries:
DataFrame: DataFrames are part of the pandas library, which is built on top of NumPy. pandas provides high-level data structures and functions designed for data analysis and manipulation.
NumPy Array: NumPy arrays are part of the NumPy library, which is a fundamental package for numerical computing in Python. NumPy provides support for multidimensional arrays, mathematical functions, random number generation, and more.
"""

if SHOW_DATAFRAMES:
    # Creating a DataFrame from a dictionary
    data = {
        'Name': ['John', 'Anna', 'Peter', 'Linda'],
        'Age': [28, 35, 23, 45],
        'City': ['Rome', 'Paris', 'London', 'Sydney']
    }

    df = pd.DataFrame(data)
    analyze_object(df)

    print(df)

    print(f"Element at '0', 'Name': {df.loc[0, 'Name']}")
    print(f"Element at '0', 'Name': {df.loc[0, 'City']}")
    print(f"Element at position (0, 1): {df.iloc[0, 1]}")
    print(f"Element at position (2, 2): {df.iloc[2, 2]}")

    # Print the names of the rows (index)
    print("Row names:")
    print(df.index)

    # Print the names of the columns
    print("\nColumn names:")
    print(df.columns)


        # Example NumPy array
    data = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

    analyze_object(data)
    print(f"npArray:\n {data} \nShape: {data.shape}" )
    for d in data:
        print(f"Iterating npArray\n : {d}")

    # Create a DataFrame from the NumPy array
    df = pd.DataFrame(data)
    analyze_object(df)
    # We can optionally add row and column names
    df.columns = ['Column1', 'Column2', 'Column3']
    df.index = ['Row1', 'Row2', 'Row3']

    # Print the DataFrame
    print(f"DataFrame:\n {df} \nShape: {df.shape}" )


    # In pandas DataFrames, when using .loc[], you first specify the row label and then the column label. 
    print(f"Element at 'Row1', 'Column1': {df.loc['Row1', 'Column1']}")
    print(f"Element at 'Row3', 'Column2': {df.loc['Row3', 'Column2']}")
    print(f"Element at position (0, 1): {df.iloc[0, 1]}")
    print(f"Element at position (2, 2): {df.iloc[2, 2]}")

    # Print the names of the rows (index)
    print("Row names:")
    print(df.index)

    # Print the names of the columns
    print("\nColumn names:")
    print(df.columns)


    # Iterate over columns
    for col_name, col_data in df.items():
        print(f'Column name: {col_name}')
        print(f'Column data:\n{col_data}')
        print('---')

        # Iterate over rows
    for idx, row in df.iterrows():
        print(f'Row index: {idx}')
        print(f'Row data:\n{row}')
        print('---')


        

    # Example list of strings
    list_of_strings = ['description 1', 'description 2', 'description 3']

    # Create a DataFrame
    df = pd.DataFrame(list_of_strings, columns=['descriptions'])

    # Print the DataFrame
    print(df)

    # Print the DataFrame
    print(f"DataFrame:\n {df} \nShape: {df.shape}" )




# Creating an array with shape (3,)
print(f"Creating a numpy array with shape (3,).\n This array has three elements along a single dimension")
arr_1d = np.array([1, 2, 3])
analyze_object(arr_1d)
print("Array:")
print(arr_1d)



# Creating an array with shape (3,1)
print(f"Creating a numpy array with shape (3,1)\n a two-dimensional array with three rows and one column. Each row contains one element, resulting in a column vector.")
arr_2d = np.array([[1],
                    [2],
                    [3]])

analyze_object(arr_2d)
print("Array:")
print(arr_2d)


# Creating an array with shape (1,3)
print(f"Creating an array with shape (1,3).\n")
array_1x3 = np.array([[1, 2, 3]])
analyze_object(array_1x3)
print(array_1x3)



# Creating an array with shape (3,0)
print(f"Creating an empty numpy array with shape (3,0).\n")
arr_2d_empty = np.empty((3, 0))
analyze_object(arr_2d_empty)
print("Array:")
print(arr_2d_empty)


# Three-dimensional array with 3 rows and 0 columns
print(f"Creating a numpy array with shape (3,3).\nTwo-dimensional array with 3 rows and 3 columns")
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
analyze_object(array_2d)
print(array_2d)
# Access the element in the second row and third column
element = array_2d[1, 2]
print("Element in the second row and third column:", element)


# Create a two-dimensional array
print(f"Creating a numpy array with shape (3,2,3).\n This indicates a three-dimensional array with 3 rows and 3 columns, where each element in the array is itself an array with 3 elements.")
array_3d = np.array(
    [[[1, 2, 3], [4, 5, 6]],
      [[7, 8, 9], [10, 11, 12]], 
      [[13, 14, 15], [16, 17, 18]]]
    )
analyze_object(array_3d)
print(array_3d)
print(f"Element at (1,1):", array_2d[1, 1])
print(f"Element at (2,2):", array_2d[2, 2]) 
print(f"Subset [1, 1, :]:{array_3d[1, 1, :]}")
print(f"Subset [2, 1, :]:{array_3d[2, 1, :]}")
print(f"Subset [:, 1, :]:\n{array_3d[:, 1, :]}")







"""
a Series is a one-dimensional labeled array. 
While a Series doesn't have column labels like a DataFrame, it can have row labels, 
which are referred to as the index of the Series. 
the index is not optional in a pandas Series. Every Series must have an index, which serves as the row labels.
If you don't explicitly specify an index when creating a Series, pandas will automatically generate a default integer index starting from 0.
While a Series can be considered a special case of a DataFrame in terms of data structure, they serve different purposes and are optimized for different types of data and operations.
you should use .iloc[] for positional indexing and .loc[] for label-based indexing.
"""

# Create a Series with row labels
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
analyze_object(s)
print("Shape:", s.shape)
# Print the Series
print(s)
print(f"The element at a: {s.loc['a']}")
print(f"The element at row 2: {s.iloc[2]}")

# Create a Series with row labels
s = pd.Series(["This is the first string", "This is the second string", "This is the third string"], index=['a', 'b', 'c'])
analyze_object(s)
print("Shape:", s.shape)
# Print the Series
print(s)
print(f"The element at a: {s.loc['a']}")
print(f"The element at row 2: {s.iloc[2]}")


# Create a TensorFlow tensor
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])

# Analyze the TensorFlow tensor
analyze_object(tensor)

# Create a TensorFlow tensor with strings
tensor_strings = tf.constant(["apple", "banana", "orange"])

# Analyze the TensorFlow tensor
analyze_object(tensor_strings)

# Create a NumPy array
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])

# Convert NumPy array to TensorFlow tensor
tensor = tf.convert_to_tensor(numpy_array)
analyze_object(tensor)

# Access a specific element
element = tensor[0, 1]  # Accesses the element at row 0, column 1
print(f"Tensor Element:\n {element.numpy()}\n")  # Convert to NumPy array for printing

# Extract a sub-tensor
sub_tensor = tensor[:, 1:]  # Extracts all rows, but only columns from index 1 onwards
print(f"Sub Tensor:\n{sub_tensor.numpy()}\n")   # Convert to NumPy array for printing

print(f"Creating a tensor with shape (3,2,3).\n This indicates a three-dimensional tensor with 3 rows and 3 columns, where each element in the array is itself an array with 3 elements.")
# Create a 3-dimensional tensor
tensor_3d = tf.constant([
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]],
    [[13, 14, 15], [16, 17, 18]]
])

analyze_object(tensor_3d)
print(tensor_3d)