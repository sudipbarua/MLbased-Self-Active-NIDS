# Initial Graphs:
### Active Learning Methods:
1. PL_RF: Passive learning random sampling method using **Random Forest algorithm**
2. PL_KNN: Passive learning random sampling method using **K-Nearest Neighbours**
3. AL_LR_US: Active learning uncertainty sampling using **Logistic regression**
4. AL_RF_US: Active learning uncertainty sampling using **Random Forest algorithm**
5. AL_KNN_US: Active learning uncertainty sampling using **Logistic regression**
6. EnS_majority: Ensemble Sampling with soft voting using "sklearn.ensemble.VotingClassifier" Library
   <br /> Estimators:
   - LR_EnS_maj: **Logistic regression**
   - RF_EnS_maj: **Random Forest algorithm**
   - KNN_EnS_maj: **K-Nearest Neighbours**
7. EnS_avg: Ensemble Sampling averaging predicted probabilities of different classification estimators
   <br /> Estimators:
   - LR_EnS_maj: **Logistic regression**
   - RF_EnS_maj: **Random Forest algorithm**
   - KNN_EnS_maj: **K-Nearest Neighbours** 
8. AL_SVM_C2H: Active Learning with SVM sampling datapoints close to the hyperplanes. Data points lying between -1 and +1 are queried to the oracle.
9. DPL_RS: Deep **passive** learning with random sampling
10. DPL_TOB: Deep **passive** learning with batch training 
11. DAL_US: Deep **active** leaning with uncertainty sampling
12. DAL_TOB: Deep **active** learning with batch training
13. DAL_CL_US: Deep **active** learning uncertainty sampling- query to oracle by clustering the data points with low confidence
---
**Sample Size: 530 (Train set: 353, Test set: 177)**

---
- Graph name: F1 score vs Number of Retrain <br />
x axis: F1 score <br />
y axis: Number of Retrains <br />
Insights:
   - Ultimate top performing models: PL_RF, EnS_majority, EnS_avg
   - EnS_majority, EnS_avg tend to overlap. Runtime is also similar.
   - Worst performing models: AL_LR_US, PL_KNN, DAL_US
![F1_vs_nRetrains_2021-12-15_15-38-47](https://user-images.githubusercontent.com/49356729/146638634-5676327c-2644-40c3-a3ce-ac53d412942a.jpg)
---
- Graph name: F1 score vs Number of queries <br />
x axis: F1 score <br />
y axis: Number of queries <br />
Insights:
   - Again EnS_majority, EnS_avg overlap
   - Naturally least queries are made by DAL_CL_US
   - Other models that had less queries are AL_RF_US and AL_KNN_US
![F1_vs_nQueries_2021-12-15_15-38-51](https://user-images.githubusercontent.com/49356729/146639645-a9f8676a-c190-43f4-af12-af1b0f7b73e4.jpg)

---
- Graph name: Number of queries vs Active learning methods
<img src="https://user-images.githubusercontent.com/49356729/146641272-b3e8460b-055a-4644-b021-4bc7f39fb115.jpg" alt="N_queries_vs_learning algorithm_2021-12-15_15-38-52" width="800" height="800">

---
- Graph name: Labeling Error of Normal Traffic vs Number of Samples <br />
x axis: Labeling Error of Normal Traffic <br />
y axis: Number of Samples <br />
Description: Continuous graph generated from FPR. FPR are calculated from predicted labels (from high confidenc of models) and actual labels at each iteration of retraining. <br />
Insights:
   - We can ignore the passive learning outputs (PL_RF, PL_KNN, DPL) since no labels are predicted in this case and hence irrellavant. 
   - AL_SVM_C2H, ensemble sampling models, AL_RF_US, AL_LR_US, AL_KNN_US has low and steady error rate.
   - CNN based models (DAL) has high error rate.
![Normal_Traffic_Error_vs_nSamples_2021-12-15_15-38-44](https://user-images.githubusercontent.com/49356729/146640034-32692a4a-5722-4101-a2cb-2cf34fb5aad4.jpg)

---
- Graph name: Labeling Error of Botnet vs Number of Samples <br />
x axis: Labeling Error of Botnet Traffic <br />
y axis: Number of Samples <br />
Description: Continuous graph generated from FNR. FNR are calculated from predicted labels (from high confidenc of models) and actual labels at each iteration of retraining. <br />
Insights:
   - We can ignore the passive learning outputs (PL_RF, PL_KNN, DPL) since no labels are predicted in this case and hence irrellavant. 
   - AL_KNN_US stays steady around 30%
   - Ensemble sampling and AL_RF_US saturate below 20%.
   - DAL_US and DAL_TOB shows best imporvement in error suppression over the time. From 70% to less than 5%
![Botnet_Labeling_Error_vs_nSamples_2021-12-15_15-38-42](https://user-images.githubusercontent.com/49356729/146640878-de27f27f-9469-4dad-91db-72d6c3f84cc3.jpg)

---
- Graph name: Labeling Errors **(Cumulative)** vs Learning algorithms <br />
x axis: Labeling Errors **(Cumulative)** <br />
y axis: Learning algorithms <br />
Description: Total counts of FP(Normal label error) and FN(Botnet) are calculated from predicted labels (from high confidenc of models) and actual labels at each iteration of retraining. <br />
Insights:
   - The total counts for AL_RF_US and CNN based models are really low because less predictions were made and more queries were done
   - Considsering both number of queries and error counts AL_RF_US is best (50 queries and 25 times normal label errors and 50 times bot label errors)
   - Again both ensemble sampling models (SK-Learn and developed) are similar.
<img src="https://user-images.githubusercontent.com/49356729/146641448-2bebaec7-f91f-40ae-8abc-9b18a59257c7.jpg" alt="Error_vs_learning algorithm_2021-12-15_15-38-54" width="800" height="600">

---

# Final results for CNN bsaed models:
---
Number of chunks: 3 (2 for train-66% and 1 for test- 34%) <br />
Initial Batch Size: 6000 <br />
Batch size: 3000 <br />

Models:
1. DPL_RS: Deep **passive** learning with random sampling
2. DPL_TOB: Deep **passive** learning with batch training 
3. DAL_US: Deep **active** leaning with uncertainty sampling
4. DAL_TOB: Deep **active** learning with batch training
5. DAL_CL_US: Deep **active** learning uncertainty sampling- query to oracle by clustering the data points with low confidence
---

- Graph name: F1 score vs Number of Retrain <br />
x axis: F1 score <br />
y axis: Number of Retrains <br />
Insights:
   - DPL_RS is best performance above 90%
   - Worst performing: DAL_TOB and DAL_CL_US above 80%
![F1_vs_nRetrains_2021-12-17_16-57-37](https://user-images.githubusercontent.com/49356729/146642828-8c1fe7c9-135e-4175-98d6-f778cff1c7ae.jpg)
---
- Graph name: F1 score vs Number of queries <br />
x axis: F1 score <br />
y axis: Number of queries <br />
Insights:
   - Naturally least queries are made by DAL_CL_US (only 1130; whereas DAL_US makes appox. 50,000 queries and DAL_TOB 75,000 queries)
![F1_vs_nQueries_2021-12-17_16-58-31](https://user-images.githubusercontent.com/49356729/146642925-3c834a81-2edb-463b-876d-d458eaf20fb6.jpg)
---
- Graph name: Number of queries vs Active learning methods
<img src="https://user-images.githubusercontent.com/49356729/146643242-031edd74-0fc4-4ced-b250-959999efc0e7.jpg" alt="N_queries_vs_learning algorithm_2021-12-15_15-38-52" width="500" height="500">

---
- Graph name: Labeling Error of Normal Traffic vs Number of Samples <br />
x axis: Labeling Error of Normal Traffic <br />
y axis: Number of Samples <br />
Description: Continuous graph generated from FPR. FPR are calculated from predicted labels (from high confidenc of models) and actual labels at each iteration of retraining. <br />
Insights:
   - We can ignore the passive learning outputs since no labels are predicted in this case and hence irrellavant. 
   - Best: DAL_US (7.5%). DAL_CL_US (18%), DAL_TOB (13.5%) 
![Normal_Traffic_Error_vs_nSamples_2021-12-17_16-57-11](https://user-images.githubusercontent.com/49356729/146643862-95629fcd-1167-4f8c-a2d6-98f2d57c9208.jpg)

---
- Graph name: Labeling Error of Botnet vs Number of Samples <br />
x axis: Labeling Error of Botnet Traffic <br />
y axis: Number of Samples <br />
Description: Continuous graph generated from FNR. FNR are calculated from predicted labels (from high confidenc of models) and actual labels at each iteration of retraining. <br />
Insights:
   - We can ignore the passive learning outputs since no labels are predicted in this case and hence irrellavant. 
   - DAL_US (14%). DAL_CL_US (12%), DAL_TOB (23%) 
![Botnet_Labeling_Error_vs_nSamples_2021-12-17_03-43-55](https://user-images.githubusercontent.com/49356729/146644907-4f28f589-0f5f-450e-9e21-21ce58ecc9ec.jpg)

---
- Graph name: Labeling Errors **(Cumulative)** vs Learning algorithms <br />
x axis: Labeling Errors **(Cumulative)** <br />
y axis: Learning algorithms <br />
Description: Total counts of FP(Normal label error) and FN(Botnet) are calculated from predicted labels (from high confidenc of models) and actual labels at each iteration of retraining. <br />
<img src="https://user-images.githubusercontent.com/49356729/146645046-fbf4c163-d41f-42f3-a4dd-7a9b4120d708.jpg" alt="Error_vs_learning algorithm_2021-12-17_16-58-52" width="600" height="600">

---

# Final results for Uncetainty Sampling bsaed models:
---
Number of chunks: 3 (2 for train-66% and 1 for test- 34%) <br />
Initial Batch Size: 6000 <br />
Batch size: 3000 <br />

Models:
1. AL_LR_US: Active learning uncertainty sampling using **Logistic regression**
2. AL_RF_US: Active learning uncertainty sampling using **Random Forest algorithm**
3. AL_KNN_US: Active learning uncertainty sampling using **Logistic regression**
---

- Graph name: F1 score vs Number of Retrain <br />
x axis: F1 score <br />
y axis: Number of Retrains <br />
Insights:
   - AL_RF achieves the best performance above 94% and AL_KNN 92%
   - Worst performing: AL_LR 74%. Reason is the convergence problem of Logistic Regression. lbfgs failed to converge and might have got stuck to a local minima. <br />
      The warning is given below:
      ```console
      ConvergenceWarning: lbfgs failed to converge (status=1):
      STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

      Increase the number of iterations (max_iter) or scale the data as shown in:
          https://scikit-learn.org/stable/modules/preprocessing.html
      Please also refer to the documentation for alternative solver options:
          https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        n_iter_i = _check_optimize_result(
       ```
 ![F1_vs_nRetrains_2021-12-19_03-56-08](https://user-images.githubusercontent.com/49356729/146672766-05863ed3-84a3-4ac8-a178-c32a919c00aa.jpg)
---
- Graph name: F1 score vs Number of queries <br />
x axis: F1 score <br />
y axis: Number of queries <br />
Insights:
   - AL_KNN_US makes the least amount of queries (6559). AL_RF_US makes 15638. 
   - Also AL_KNN_US has the more steady F1 score graph. Whereas AL_RF_US graph has much fluctuations. 
   - AL_LR_US makes 130,000 queries approx. But can't achieve more than 75% of F1 score.

![F1_vs_nQueries_2021-12-19_03-56-33](https://user-images.githubusercontent.com/49356729/146673048-8a5e483f-bcbb-4984-a69e-7bddf4a2fc95.jpg)
---
- Graph name: Number of queries vs Active learning methods
<img src="https://user-images.githubusercontent.com/49356729/146673080-f68e28db-defb-4b07-9892-57f8b678c247.jpg" alt="N_queries_vs_learning algorithm_2021-12-19_03-56-37" width="500" height="500">

---
- Graph name: Labeling Error of Normal Traffic vs Number of Samples <br />
x axis: Labeling Error of Normal Traffic <br />
y axis: Number of Samples <br />
Description: Continuous graph generated from FPR. FPR are calculated from predicted labels (from high confidenc of models) and actual labels at each iteration of retraining. <br />
Insights:
   - AL_KNN_US and AL_RF_US has 4% and 6% Labeling error rates respectively.
   - AL_LR_US has 14% error rate.

![Normal_Traffic_Error_vs_nSamples_2021-12-19_03-55-48](https://user-images.githubusercontent.com/49356729/146673400-1740f6ed-9686-4415-9f7b-8ff528a95107.jpg)
---
- Graph name: Labeling Error of Botnet vs Number of Samples <br />
x axis: Labeling Error of Botnet Traffic <br />
y axis: Number of Samples <br />
Description: Continuous graph generated from FNR. FNR are calculated from predicted labels (from high confidenc of models) and actual labels at each iteration of retraining. <br />
Insights:
   - AL_KNN_US and AL_RF_US has below 1% of Labeling error rate.
   - AL_LR_US has 14% error rate. 

![Botnet_Labeling_Error_vs_nSamples_2021-12-18_20-43-22](https://user-images.githubusercontent.com/49356729/146673484-14a21144-aa07-4a8f-ac19-68c5bf2b4ee0.jpg)
---
- Graph name: Labeling Errors **(Cumulative)** vs Learning algorithms <br />
x axis: Labeling Errors **(Cumulative)** <br />
y axis: Learning algorithms <br />
Description: Total counts of FP(Normal label error) and FN(Botnet) are calculated from predicted labels (from high confidenc of models) and actual labels at each iteration of retraining. <br />

<img src="https://user-images.githubusercontent.com/49356729/146673729-d363da25-28f3-493e-833b-b0e48d40ec7a.jpg" alt="Error_vs_learning algorithm_2021-12-19_03-56-56" width="600" height="600">

---
# Final results for Uncetainty Sampling bsaed model for "modified" Logistic Regression model:
---
Number of chunks: 3 (2 for train-66% and 1 for test- 34%) <br />
Initial Batch Size: 6000 <br />
Batch size: 3000 <br />
As 'lbfgs' solver has convergence issues, 'newton-cg' has been used instead to encounter the problem.<br />
Now the ultimate **F1 score = 75%** but the active learning model is the **fastest**. <br /> 
Hyper partameter setting: C=10, penalty='l2', solver='newton-cg'

---
- Graph name: F1 score vs Number of Retrain <br />
x axis: F1 score <br />
y axis: Number of Retrains <br />
![F1_vs_nRetrains_2021-12-19_12-41-30](https://user-images.githubusercontent.com/49356729/146674038-f8270f38-7f50-4f8e-8cda-11118b87296f.jpg)
---
- Graph name: F1 score vs Number of queries <br />
x axis: F1 score <br />
y axis: Number of queries <br />
Total number of queries: 96,301.
![F1_vs_nQueries_2021-12-19_12-41-31](https://user-images.githubusercontent.com/49356729/146674073-c907d452-f7eb-47a1-a871-15f2df31b621.jpg)
---
- Graph name: Labeling Error of Normal Traffic vs Number of Samples <br />
x axis: Labeling Error of Normal Traffic <br />
y axis: Number of Samples <br />
Description: Continuous graph generated from FPR. FPR are calculated from predicted labels (from high confidenc of models) and actual labels at each iteration of retraining. <br />
![Normal_Traffic_Error_vs_nSamples_2021-12-19_12-41-29](https://user-images.githubusercontent.com/49356729/146674214-e0b6f9bb-2264-4b01-b5b0-cebe6245648f.jpg)
---
- Graph name: Labeling Error of Botnet vs Number of Samples <br />
x axis: Labeling Error of Botnet Traffic <br />
y axis: Number of Samples <br />
Description: Continuous graph generated from FNR. FNR are calculated from predicted labels (from high confidenc of models) and actual labels at each iteration of retraining. <br />
![Botnet_Labeling_Error_vs_nSamples_2021-12-19_12-41-28](https://user-images.githubusercontent.com/49356729/146674213-9673006e-2a83-441e-8c03-88fb66ce7262.jpg)
---
- Graph name: Labeling Errors **(Cumulative)** vs Learning algorithms <br />
x axis: Labeling Errors **(Cumulative)** <br />
y axis: Learning algorithms <br />
Description: Total counts of FP(Normal label error) and FN(Botnet) are calculated from predicted labels (from high confidenc of models) and actual labels at each iteration of retraining. <br />
<img src="https://user-images.githubusercontent.com/49356729/146674223-a932e4c3-32ae-4937-a7d4-2038a65898ac.jpg" alt="Error_vs_learning algorithm_2021-12-19_12-41-33" width="500" height="500">

---
