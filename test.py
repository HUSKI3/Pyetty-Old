
def read_line(filename: "Name of the file to read", start: "line to start at", end: "line to end at"):
    # Try to load 
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            # We want to check if we are in bounds
            if not (start > len(lines)) and not (end > len(lines)-1):
                lines = lines[start:end]
            else:
                print(f"Failed to cut lines, out of range. Got: start={start} end={end} but maxlength={len(lines)}")
                return
        print(lines)
    # If file does not exist, catch the exception and output filenotfound
    except FileNotFoundError:
        print(f"Failed to load file [{filename}], does it exist?")
        return 

# Lines start at 0
read_line("main.ppy", 0, 445)