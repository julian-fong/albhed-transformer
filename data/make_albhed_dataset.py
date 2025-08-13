import pandas as pd
path = "albhed.txt"

try:
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
except FileNotFoundError:
    print(f"Error: The file '{path}' was not found.")
    exit()
    
translations = []
for i, line in enumerate(lines):
    if line.strip() == "\n" or line.strip() == "":
        continue
    elif ":" in line:
        text = line.split(":")[1].strip()
        splitted_line = text.split("(")
        albhed = splitted_line[0].strip()
        english = splitted_line[1].strip().replace(")", "")
    else:
        splitted_line = line.split("(")
        albhed = splitted_line[0].strip()
        english = splitted_line[1].strip().replace(")", "")
    translations.append({"albhed": albhed, "english": english})
    
df = pd.DataFrame(translations)
df.to_csv("translations.csv", index=False, encoding="utf-8")