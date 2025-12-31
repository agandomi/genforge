from genforge.spfp.spfp_partition import SPFPPartitioner
import pandas as pd
import numpy as np

xtr = pd.read_csv("ARWPM_Normalized_x_train.csv", header=0).to_numpy()
ytr = pd.read_csv("ARWPM_Normalized_y_train.csv", header=0).to_numpy()

xval = pd.read_csv("ARWPM_Normalized_x_validation.csv", header=0).to_numpy()
yval = pd.read_csv("ARWPM_Normalized_y_validation.csv", header=0).to_numpy()

xts = pd.read_csv("ARWPM_Normalized_x_test.csv", header=0).to_numpy()
yts = pd.read_csv("ARWPM_Normalized_y_test.csv", header=0).to_numpy()

X_all = np.vstack((xtr, xval, xts))
Y_all = np.vstack((ytr, yval, yts)).ravel()

# X: (n_samples, n_features), y: (n_samples,)
spfp = SPFPPartitioner(
    n_groups=2,
    n_bins=10,
    objective="micor",   # 'mi', 'cor', or 'micor'
    override=False,
    backward=False,
    perfs=0.10,
    remp=0.60,
    random_state=7,
    verbose=1
)

spfp.fit(X_all, Y_all)          # optional: pass X_test, y_test for reporting
groups = spfp.partition()
partitions = spfp.get_partitions()  # [[fidx...], [fidx...], ...]


max_len = max(len(g) for g in partitions)
for g in partitions:
    g.extend([None] * (max_len - len(g)))  # fill shorter groups with blanks

# Step 2. Create DataFrame (each group = column)
df = pd.DataFrame({f"Group # {i+1}": partitions[i] for i in range(len(partitions))})

# Step 3. Write to Excel
df.to_excel("SPFP_partitions.xlsx", index=False)