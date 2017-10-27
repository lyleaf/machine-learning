%%bq query --name roc_prediction
SELECT 
image_url, 
CASE WHEN predicted = 'cancer' THEN predicted_prob ELSE 1-predicted_prob END AS prob,
CASE WHEN target = 'cancer' THEN 1 ELSE 0 END AS label
FROM dermNet.eval_results_local_v2;

%%bq query --name roc_prediction_small
SELECT 
image_url, 
CASE WHEN predicted = 'cancer' THEN predicted_prob ELSE 1-predicted_prob END AS prob,
CASE WHEN target = 'cancer' THEN 1 ELSE 0 END AS label
FROM dermNet.eval_results_local;

import google.datalab.bigquery as bq
from sklearn import metrics
import matplotlib.pyplot as plt

df = roc_prediction.execute(output_options=bq.QueryOutput.dataframe()).result()
df_small = roc_prediction_small.execute(output_options=bq.QueryOutput.dataframe()).result()


matrix = df.as_matrix()
label = matrix[:,-1]
prob = matrix[:,1]
myroundedprobs = [ round(elem,2) for elem in prob]

matrix_small = df_small.as_matrix()
label_small = matrix_small[:,-1]
prob_small = matrix_small[:,1]
myroundedprobs_small = [ round(elem,2) for elem in prob_small]


fpr, tpr, thresholds = metrics.roc_curve(label, myroundedprobs, pos_label=1)
fpr_small, tpr_small, thresholds_small = metrics.roc_curve(label_small, myroundedprobs_small, pos_label=1)


roc_auc = metrics.auc(fpr, tpr)
roc_auc_small = metrics.auc(fpr_small, tpr_small)

print roc_auc
print roc_auc_small

plt.title('ROC - Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'green', label='AUC = %0.2f'% roc_auc)
plt.plot(fpr_small, tpr_small, 'blue', label='AUC = %0.2f'% roc_auc_small)

plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

# create the axis of thresholds (scores)
ax2 = plt.gca().twinx()
ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')
ax2.set_ylabel('Threshold',color='g')
ax2.set_ylim([thresholds[-1],thresholds[0]])
ax2.set_xlim([fpr[0],fpr[-1]])
plt.show()

label_binary = [True if l == 1 else False for l in label]
metrics.average_precision_score(label_binary, prob)
