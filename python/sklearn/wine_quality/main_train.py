"""
Standalone main for Databricks jobs - for "python_file" in "spark_python_task".
"""

import sys
from wine_quality import train

if __name__ == "__main__":
    train.main(sys.argv[1:])
