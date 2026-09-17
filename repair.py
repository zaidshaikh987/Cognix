import os

filepath = "dashboard/static/index.html"
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Fix the escaped backticks starting template literals
content = content.replace(r"\`", "`")

# Fix the escaped dollar signs in template literals
content = content.replace(r"\${", "${")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print("index.html JS syntax repaired.")
