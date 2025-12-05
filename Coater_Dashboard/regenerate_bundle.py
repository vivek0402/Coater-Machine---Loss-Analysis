import sys
import os

# Ensure current directory is in path
sys.path.append(os.getcwd())

import model_connection

print(f"CombinedModel module: {model_connection.CombinedModel.__module__}")
model_connection.main()
print("Done.")
