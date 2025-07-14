import pandas as pd #'pd' is the standard alias for pandas
import matplotlib.pyplot as plt # 'plt' is the standard alias
import seaborn as sns # 'sns' is the standard alias

# --- Part 1: Reuse our logic from Day 2 to extract error data ---
# (In a real project, we would make this a reusable function)

log_lines=[]
with open('server.log', 'r') as file:
    log_lines=file.readlines()

error_reports=[]
for line in log_lines:
    if 'ERROR' in line:
        parts=line.split(' ', 3)
        message=parts[3]
        # Extract the module name (e.g., '[AuthService]')
        module=message[message.find("[")+1:message.find("]")]

        error_details = {
            'timestamp': pd.to_datetime(parts[0]+ ' ' + parts[1]),
            'module': module,
            'message': message.strip()
        }
        error_reports.append(error_details)

# --- Part 2: Load data into a Pandas DataFrame for analysis ---
# This is where the magic begins.

error_df = pd.DataFrame(error_reports)
print("--- Error Data Loaded into a DataFrame ---")
print(error_df)
print("\n" + "="*40 + "\n")

# --- Part 3: Perform the analysis ---

# 1. Calculate the average time between failures
# The .diff() function calculates the difference between an element and the one before it.
# This uses NumPy's power under the hood!

time_between_errors = error_df['timestamp'].diff().mean()
# 2. Count errors by module
# The .groupby() function is one of the most powerful tools in Pandas.

errors_by_module = error_df.groupby('module').size().reset_index(name='error_count')
# --- Part 4: Display the results ---
print("--- Analysis Results ---")
print(f"Average time between failures: {time_between_errors}")
print("\nBreakdown of errors by module:")
print(errors_by_module)

# --- Part 5: Visualize the results ---
print("\n--- Generating Visuals ---")

# Set a nice style for the plot
sns.set_theme(style="whitegrid")
# Create the bar plot using Seaborn
plt.figure(figsize=(10,6)) # Set the figure size to make it more readable
barplot = sns.barplot(x='error_count', y='module', data=errors_by_module)

# Add title and labels for clarity
plt.title('Total Errors by Software Module')
plt.xlabel('Number of Errors')
plt.ylabel('Module')

# Save the plot to a file
# This is what you would send to your manager
plt.savefig("module_error_description.png")
print("Chart saved to 'module_error_distribution.png'")

# Display the plot on your screen
plt.show()