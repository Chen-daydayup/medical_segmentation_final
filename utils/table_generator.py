import pandas as pd
import os
from config import cfg

def save_comparison_table(results, filename):
    df = pd.DataFrame(results)
    save_path = os.path.join(cfg.SAVE_DIR, filename)
    df.to_csv(save_path, index=False)
    print(f"✅ 已保存: {save_path}")
