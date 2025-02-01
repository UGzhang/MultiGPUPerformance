#!/usr/bin/python

# Script to write out obstacle coordinates

n = 10240

# Open a file to write the output
with open("output.txt", "w") as file:
    # Write out 'bottom rail'
    for x in range(0, n):
        file.write(f"{x} 0 1\n")

    # Write out 'top rail'
    for x in range(0, n):
        file.write(f"{x} {n-1} 1\n")

    # Write out 'left rail'
    for x in range(0, n):
        file.write(f"0 {x} 1\n")

    # Write out 'right rail'
    for x in range(0, n):
        file.write(f"{n-1} {x} 1\n")