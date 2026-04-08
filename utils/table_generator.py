import pandas as pd
import os
from config import cfg

def save_comparison_table(results):
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(cfg.SAVE_DIR, "comparison.csv"), index=False)
    print("Comparison table saved to results/comparison.csv")