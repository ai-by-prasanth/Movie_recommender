# -- Part 1 Print errors from server.log file 
print("Reading log file...")
with open('server.log', 'r') as file:
    log_lines=file.readlines()

# the with open(..) is a best practice to open the file
# It handles opening , crucially and automatically closing the file.
# .readlines() reads all the lines from the file into a list, where each line is a item

#Let's see what we have 
print("log files  loaded into a list")
print(log_lines)
print("-" * 20 ) # a simple operator 

#Part 2 processing the lines to find errors 
print("Searching for error level messages...")
error_reports = [] #start with an empty list to hold our structured data 

for line in log_lines:
    #we can use 'in' keyword to search for a substring in the line
    if 'ERROR' in line:
        # This line is an error! Let's structure it.
        # We'll use .split() to break the line into pieces.
        # Let's split it by spaces, but only the first 3, to group the message nicely.
        parts = line.split(' ', 3)
        # Now we create a dictionary for this specific error
        error_details = {
            'timestamp':  parts[0] + ' ' + parts[1],
            'level': parts[2],
            'message': parts[3].strip() # .strip() removes whitespace and newlines 
        }
        # Add our new dictionary to the list of error reports
        error_reports.append(error_details)

print("Found {} errors.".format(len(error_reports)))
print("Structured error data:")
print(error_reports)
print("-" * 20 )

# --- Part 3: Write the summary to a new file ---
print("Writing summary report to 'error_summary.txt'...")

with open('error_summary.txt', 'w') as summary_file:
    for error in error_reports:
        # Let's write a clean, readable line to our report
        summary_file.write(f"Time: {error['timestamp']}, Message:{error['message']}\n")

print("Done!")
