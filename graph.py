#%%
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

#%%
# log_path = "./log"
result_path = "result/2022-09-25-23-52/"
graph_path = "graph"
graph_path = os.path.join(result_path, graph_path)
print(graph_path)
os.makedirs(graph_path, exist_ok=True)
# graph_path = "./graph/"
with open(result_path + '/train.pkl', 'rb') as f:
    log = pickle.load(f)

#%%
plt.figure()
plt.plot(log['train_loss'], label='train loss')
plt.plot(log['test_loss'], label='test loss')
plt.legend()
plt.savefig(graph_path + '/loss.png', bbox_inches='tight', pad_inches=0.03, dpi=600)

# %%
plt.figure()
plt.plot(log['train_acc'], label='train acc')
plt.plot(log['test_acc'], label='test acc')
plt.legend()
plt.savefig(graph_path + '/acc.png', bbox_inches='tight', pad_inches=0.03, dpi=600)

# %%
