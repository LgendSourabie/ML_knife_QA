import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


model_index = ['Classification', 'Regression']

df_model_performance = pd.DataFrame(index=model_index, columns=['Accuracy', 'Precision', 'Recall'])

acc_class, acc_reg = 0.7286,0.6992
rec_class, rec_reg= 0.5899,0.5711
prec_class, prec_reg = 0.7752,0.7264

df_model_performance['Accuracy'] = [acc_class, acc_reg]
df_model_performance['Precision'] = [rec_class, rec_reg]
df_model_performance['Recall'] = [prec_class, prec_reg]

ax = df_model_performance.plot.bar()
ax.legend(
    ncol=len(model_index),
    bbox_to_anchor=(0, 1),
    loc='lower left',
    prop={'size': 14}
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')

plt.tight_layout()
plt.savefig('DL_comparison.png', dpi=300)