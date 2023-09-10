file_path = './data/Syslog/syslog'

# Initialize a line count variable
line_count = 0

# Open the file and iterate through its lines
with open(file_path, 'r') as file:
    for line in file:
        line_count += 1

# Print the total number of lines
print(f"Total number of lines: {line_count}")
