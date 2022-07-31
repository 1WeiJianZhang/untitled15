import pandas as pd
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
SEED_NUM = 10
ENV = "20_jobs"
if __name__ == '__main__':
        algo = ["per" + ENV, "basic" + ENV,
                "ddqn" + ENV]
        data = []
        for i in range(len(algo)):
            for seed in range(SEED_NUM):
                file = os.path.join(os.path.join(algo[i], algo[i] + "_s" + str(seed * 10)), "progress.txt")

                pd_data = pd.read_table(file)
                pd_data.insert(len(pd_data.columns), "Unit", seed)
                pd_data.insert(len(pd_data.columns), "Condition", algo[i])

                data.append(pd_data)
        data = pd.concat(data, ignore_index=True)
        sns.set(style="darkgrid", font_scale=1.5)
        sns.tsplot(data=data, time="episode", value="makespan", condition="Condition", unit="Unit", ci='sd')
        # 数据大时使用科学计数法
        plt.legend(loc='best').set_draggable(True)
        plt.tight_layout(pad=0.5)
        plt.show()









