import csv


# Define a function to map skill levels to numbers
def map_skill(skill):
    if skill == "low":
        return 1
    elif skill == "medium":
        return 2
    elif skill == "high":
        return 3
    else:
        return None


# Open the input and output files
with open("data/mock.csv", "r") as input_file, open(
    "data/output.csv", "w", newline=""
) as output_file:
    # Create a CSV reader and writer
    reader = csv.reader(input_file)
    writer = csv.writer(output_file)

    # Loop through each row in the input file
    for row in reader:
        # Remove the first column (name)
        row.pop(0)

        # Map the skill level to a number
        skill = map_skill(row[1])
        if skill is None:
            continue

        # Replace the skill level with the mapped number
        row[1] = skill

        # Write the modified row to the output file
        writer.writerow(row)
