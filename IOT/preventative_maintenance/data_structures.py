import tensorflow as tf
import numpy as np
import pandas as pd
import torch
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Layer
import tensorflow  as tf




SHOW_DATAFRAMES=False
SHOW_NUMPY_ARRAYS=False
SHOW_SERIES=False
SHOW_TENSORS=True

def analyze_object(obj,name=""):
    print("------------------------ Object Analysis Begins for {name} --------------------------")
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
    elif isinstance(obj, torch.Tensor):
        print("Object is a PyTorch tensor")
        print("Shape:", obj.shape)
        print("Dimensions in this tensor: ", obj.dim())
        print("Length of PyTorch tensor:", obj.size(0))
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

    print(f"Extracting a subset from a dataframe.\n")

    # Example DataFrame
    data = {
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    }
    df = pd.DataFrame(data)
    print(data)
    # Extracting a subset of columns
    subset = df[['A', 'C']]  # Selecting columns 'A' and 'C'
    print("Subset\n")
    print(subset)

    print("We can also select a subset of a dataframe that consists of every second row.")

    # Example DataFrame
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [6, 7, 8, 9, 10],
        'C': [11, 12, 13, 14, 15]
    }
    df = pd.DataFrame(data)

    print(df)
    # Selecting every second row
    every_second_row = df.iloc[::2]

    print("every_second_row\n")
    print(every_second_row)


if SHOW_SERIES:    
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

if SHOW_NUMPY_ARRAYS:
    # Creating an array with shape (3,)
    print(f"Creating a numpy array with shape (3,).\n This array has three elements along a single dimension")
    arr_1d = np.array([1, 2, 3])
    analyze_object(arr_1d,"array_1d")
    print("Array:")
    print(arr_1d)

    # Creating an array with shape (3,1)
    print(f"Creating a numpy array with shape (3,1)\n a two-dimensional array with three rows and one column. Each row contains one element, resulting in a column vector.")
    arr_2d = np.array([[1],
                        [2],
                        [3]])

    analyze_object(arr_2d,"array_2d")
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




    # Example NumPy array
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Extracting a subset of columns
    subset = array[:, [0, 2]]  # Selecting columns 0 and 2

    print(f"subset: array[:, [0, 2]]\n {array[:, [0, 2]]}")

    # Example NumPy array
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    # Selecting every second row
    every_second_row = array[::2]

    print(f"every_second_row = \n{every_second_row}")


    # Create a sample 2D NumPy array
    arr = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

    # Calculate the mean of each row
    row_means = np.mean(arr, axis=1)

    # Print the mean of each row
    print("Mean of each row:")
    for i, mean in enumerate(row_means):
        print(f"Row {i+1}: {mean}")

    # Create a sample 2D NumPy array
    arr = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

    # Filter rows where the sum of elements is greater than 10
    filtered_rows = arr[arr.sum(axis=1) > 10]

    # Filter elements greater than 5
    filtered_elements = arr[arr > 5]

    # Print the filtered rows and elements
    print("Filtered rows where the sum of elements is greater than 10:")
    print(filtered_rows)

    print("\nFiltered elements greater than 5:")
    print(filtered_elements)  

    # Create a sample 2D NumPy array
    arr = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

    # Filter columns where the sum of elements is greater than 10
    filtered_columns = arr[:, arr.sum(axis=0) > 10]

    # Filter elements in specific columns greater than 5
    filtered_elements = arr[:, (arr[:, 1] > 5) | (arr[:, 2] > 5)]

    # Print the filtered columns and elements
    print("Filtered columns where the sum of elements is greater than 10:")
    print(filtered_columns)

    print("\nFiltered elements in specific columns greater than 5:")
    print(filtered_elements)  



if SHOW_TENSORS:
    print(f"Creating a tensorflow tensor with shape (2,3).\n")
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
    print(f"Creating a tensorflow tensor with shape (3,2,3).\n This indicates a three-dimensional tensor with 3 rows and 3 columns, where each element in the array is itself an array with 3 elements.")
    # Create a 3-dimensional tensor
    tensor_3d = tf.constant([
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
        [[13, 14, 15], [16, 17, 18]]
    ])
    analyze_object(tensor_3d)
    print(tensor_3d)

    print(f"Creating a TensorFlow tensor with shape (2,3).\n")
    # Create a PyTorch tensor
    tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
    analyze_object(tensor)
    print(tensor)
    # Accessing individual elements
    print("Element at position (0, 0):", tensor[0, 0])  # Accessing the element in the first row and first column
    print("Element at position (1, 2):", tensor[1, 2])  # Accessing the element in the second row and third column


    print(f"Creating a pyTorch tensor with shape (2,3).\n")
    # Create a PyTorch tensor
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    analyze_object(tensor)
    print(tensor)
    # Accessing individual elements
    print("Element at position (0, 0):", tensor[0, 0])  # Accessing the element in the first row and first column
    print("Element at position (1, 2):", tensor[1, 2])  # Accessing the element in the second row and third column

    # Slicing to access subsets of the tensor
    print("First row:", tensor[0])    # Accessing the first row
    print("Second column:", tensor[:, 1])  # Accessing the second column

    print(f"Creating a torch tensor subset of every second row\n")


    # Example PyTorch tensor
    print(f"Creating a tensorflow tensor subset of every second row\n")
    tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    print(tensor)
    # Selecting every second row
    every_second_row = tensor[::2]
    print("Every_second_row\n")
    print(every_second_row)


    # Example PyTorch tensor
    print(f"Creating a torch tensor subset of every second row\n")
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    print(tensor)
    # Selecting every second row
    every_second_row = tensor[::2]
    print("Every_second_row\n")
    print(every_second_row)


    print(f"Creating a Tensorflow tensor subset of columns\n")
    # Example PyTorch tensor
    tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    print(tensor)
    # Extracting columns 0 and 2
    subset_tensor = tf.gather(tensor, [0, 2], axis=1)
    print("\nSubset Tensor (columns 0 and 2):")
    print(subset_tensor)



    print(f"Creating a pytorch tensor subset of columns\n")
    # Example PyTorch tensor
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    print(tensor)
    # Selecting a subset of columns
    subset = tensor[:, [0, 2]]  # Selecting columns 0 and 2
    print("Pytorch Column Subset: \n")
    print(subset)


    x = np.arange(20).reshape(2, 2, 5)
    y = np.arange(20, 30).reshape(2, 1, 5)
    analyze_object(x)
    analyze_object([x,y])
    # When the inputs are NumPy arrays, TensorFlow automatically converts them to tensors internally.
    tf.keras.layers.concatenate([x, y], axis=1)  # (2, 3, 5)


    # Create two 3D tensors
    x = np.arange(24).reshape(2, 3, 4)
    y = np.arange(24, 36).reshape(1, 3, 4)

    print(f"x = {x}")
    print(f"y = {y}")

    # Concatenate tensors along axis 0
    concatenated_axis_0 = tf.keras.layers.concatenate([x, y], axis=0)
    print("Concatenated along axis 0:", concatenated_axis_0.shape)  # (3, 3, 4)

    # Concatenate tensors along axis 1
    # Create two 3D tensors
    x = np.arange(24).reshape(2, 3, 4) # np.arange(24): This part generates an array containing integers from 0 to 23. The 
    # .reshape(2, 3, 4): After generating the array of integers from 0 to 23, we reshape it into a 3-dimensional array with shape (2, 3, 4). This means the array will have:
    # 2 elements along the first dimension
    # 3 elements along the second dimension
    # 4 elements along the third dimension
    # Original array (flattened):
    # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]

    # Reshaped array (3D):
    # [
    #  [[ 0  1  2  3]
    #   [ 4  5  6  7]
    #   [ 8  9 10 11]]
    
    #  [[12 13 14 15]
    #   [16 17 18 19]
    #   [20 21 22 23]]
    # ]

    # For a given shape, the size of the array is the product of all the numbers in the shape tuple.
    # Conversely, for a given size, there can be multiple valid shapes that can accommodate that size.
    y = np.arange(24, 48).reshape(2, 3, 4) # generates an array containing integers from 24 to 35

    # Concatenate tensors along axis 1
    # Create two 3D tensors
    x = np.arange(24).reshape(2, 3, 4)
    y = np.arange(24, 48).reshape(2, 3, 4)  # Corrected shape



    print(f"Original tensor x, shape: {x.shape} size {x.size}")  # (2, 3, 4)
    print(f"Original tensor y, shape: {y.shape} size {y.size}")  # (2, 3, 4)

    # Concatenate tensors along axis 0
    concatenated_axis_0 = tf.keras.layers.concatenate([x, y], axis=0)
    print("Concatenated tensor shape along axis 0:", concatenated_axis_0.shape)  # (2, 6, 4)
    print(concatenated_axis_0)

    # Concatenate tensors along axis 2
    concatenated_axis_1 = tf.keras.layers.concatenate([x, y], axis=1)
    print("Concatenated tensor shape along axis 1:", concatenated_axis_1.shape)  # (2, 6, 4)
    print(concatenated_axis_1)

    # Concatenate tensors along axis 3
    concatenated_axis_2 = tf.keras.layers.concatenate([x, y], axis=2)
    print("Concatenated tensor shape along axis 2:", concatenated_axis_2.shape)  # (2, 6, 4)
    print(concatenated_axis_2)