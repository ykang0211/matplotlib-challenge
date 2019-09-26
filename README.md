
### Include 3 observations about the results of the study. Use the visualizations you generated from the study data as the basis for your observations.
1. Capomulin is the best treatment among others, shows reduced tumor volume, less metastatic spread, and high survival rate
2. Infubinol shows less metatstic spread and lower tumor volume than Ketapril and Placebo, not significantly, but surviving rate is very low.
3. However, both Ketapril and Infubinol show inefficiency as Placebo

Creating a scatter plot that shows how the tumor volume changes over time for each treatment.
Creating a scatter plot that shows how the number of metastatic (cancer spreading) sites changes over time for each treatment.
Creating a scatter plot that shows the number of mice still alive through the course of treatment (Survival Rate)
Creating a bar graph that compares the total % tumor volume change for each drug across the full 45 days.


```python
# %matplotlib notebook
```


```python
# dependencies
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import sem

# hide warning messages in notebook
import warnings
warnings.filterwarnings("ignore")

# import and read csv
mouse_drug_data_to_load = "data/mouse_drug_data.csv"
clinical_trial_data_to_load = "data/clinicaltrial_data.csv"

mouse_df = pd.read_csv(mouse_drug_data_to_load)
clinical_df = pd.read_csv(clinical_trial_data_to_load)

# merge two data
mouse_clinical = pd.merge(clinical_df, mouse_df, on = "Mouse ID", how = "left")

# sort values
mouse_clinical = mouse_clinical.sort_values(["Timepoint", "Tumor Volume (mm3)", "Metastatic Sites"])

# display data tablel for preview
mouse_clinical = mouse_clinical[["Mouse ID", "Timepoint", "Tumor Volume (mm3)", "Metastatic Sites", "Drug"]]

mouse_clinical.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mouse ID</th>
      <th>Timepoint</th>
      <th>Tumor Volume (mm3)</th>
      <th>Metastatic Sites</th>
      <th>Drug</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>b128</td>
      <td>0</td>
      <td>45.0</td>
      <td>0</td>
      <td>Capomulin</td>
    </tr>
    <tr>
      <td>1</td>
      <td>f932</td>
      <td>0</td>
      <td>45.0</td>
      <td>0</td>
      <td>Ketapril</td>
    </tr>
    <tr>
      <td>2</td>
      <td>g107</td>
      <td>0</td>
      <td>45.0</td>
      <td>0</td>
      <td>Ketapril</td>
    </tr>
    <tr>
      <td>3</td>
      <td>a457</td>
      <td>0</td>
      <td>45.0</td>
      <td>0</td>
      <td>Ketapril</td>
    </tr>
    <tr>
      <td>4</td>
      <td>c819</td>
      <td>0</td>
      <td>45.0</td>
      <td>0</td>
      <td>Ketapril</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Store the Mean Tumor Volume Data Grouped by Drug and Timepoint 
# tumor_volume_mean = mouse_clinical.groupby(["Drug", "Timepoint"]).mean()
# tumor_volume_mean = mouse_clinical.groupby(["Drug", "Timepoint"])["Tumor Volume(mm3)"].mean()
# tumor_volume_mean = mouse_clinical.groupby(["Drug", "Timepoint"]).mean()["Tumor Volume (mm3)"]

tumor_volume_mean = mouse_clinical.groupby(["Drug", "Timepoint"])["Tumor Volume (mm3)"]
tumor_volume_mean_df = tumor_volume_mean.mean()

# Convert to DataFrame
tumor_volume_mean_df = tumor_volume_mean_df.reset_index()

# Preview DataFrame
tumor_volume_mean_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Drug</th>
      <th>Timepoint</th>
      <th>Tumor Volume (mm3)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Capomulin</td>
      <td>0</td>
      <td>45.000000</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Capomulin</td>
      <td>5</td>
      <td>44.266086</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Capomulin</td>
      <td>10</td>
      <td>43.084291</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Capomulin</td>
      <td>15</td>
      <td>42.064317</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Capomulin</td>
      <td>20</td>
      <td>40.716325</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>95</td>
      <td>Zoniferol</td>
      <td>25</td>
      <td>55.432935</td>
    </tr>
    <tr>
      <td>96</td>
      <td>Zoniferol</td>
      <td>30</td>
      <td>57.713531</td>
    </tr>
    <tr>
      <td>97</td>
      <td>Zoniferol</td>
      <td>35</td>
      <td>60.089372</td>
    </tr>
    <tr>
      <td>98</td>
      <td>Zoniferol</td>
      <td>40</td>
      <td>62.916692</td>
    </tr>
    <tr>
      <td>99</td>
      <td>Zoniferol</td>
      <td>45</td>
      <td>65.960888</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 3 columns</p>
</div>




```python
 # Store the Standard Error of Tumor Volumes Grouped by Drug and Timepoint
tumor_volume_sem = mouse_clinical.groupby(["Drug", "Timepoint"])["Tumor Volume (mm3)"]
tumor_volume_sem_df = tumor_volume_sem.sem()

# Convert to DataFrame
tumor_volume_sem_df = tumor_volume_sem_df.reset_index()

# Preview DataFrame
tumor_volume_sem_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Drug</th>
      <th>Timepoint</th>
      <th>Tumor Volume (mm3)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Capomulin</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Capomulin</td>
      <td>5</td>
      <td>0.448593</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Capomulin</td>
      <td>10</td>
      <td>0.702684</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Capomulin</td>
      <td>15</td>
      <td>0.838617</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Capomulin</td>
      <td>20</td>
      <td>0.909731</td>
    </tr>
  </tbody>
</table>
</div>




```python
 # Minor Data Munging to Re-Format the Data Frames
# data_munging = tumor_volume_mean_df.set_index(["Drug", "Timepoint", "Tumor Volume (mm3)"], drop = True).unstack("Drug")
data_munging = tumor_volume_mean_df.pivot_table("Tumor Volume (mm3)", ["Timepoint"], "Drug")

data_munging_sem = tumor_volume_sem_df.pivot_table("Tumor Volume (mm3)", ["Timepoint"], "Drug")

# Preview that Reformatting worked
# tumor_volume_sem_df
data_munging.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Ceftamin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Naftisol</th>
      <th>Placebo</th>
      <th>Propriva</th>
      <th>Ramicane</th>
      <th>Stelasyn</th>
      <th>Zoniferol</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
    </tr>
    <tr>
      <td>5</td>
      <td>44.266086</td>
      <td>46.503051</td>
      <td>47.062001</td>
      <td>47.389175</td>
      <td>46.796098</td>
      <td>47.125589</td>
      <td>47.248967</td>
      <td>43.944859</td>
      <td>47.527452</td>
      <td>46.851818</td>
    </tr>
    <tr>
      <td>10</td>
      <td>43.084291</td>
      <td>48.285125</td>
      <td>49.403909</td>
      <td>49.582269</td>
      <td>48.694210</td>
      <td>49.423329</td>
      <td>49.101541</td>
      <td>42.531957</td>
      <td>49.463844</td>
      <td>48.689881</td>
    </tr>
    <tr>
      <td>15</td>
      <td>42.064317</td>
      <td>50.094055</td>
      <td>51.296397</td>
      <td>52.399974</td>
      <td>50.933018</td>
      <td>51.359742</td>
      <td>51.067318</td>
      <td>41.495061</td>
      <td>51.529409</td>
      <td>50.779059</td>
    </tr>
    <tr>
      <td>20</td>
      <td>40.716325</td>
      <td>52.157049</td>
      <td>53.197691</td>
      <td>54.920935</td>
      <td>53.644087</td>
      <td>54.364417</td>
      <td>53.346737</td>
      <td>40.238325</td>
      <td>54.067395</td>
      <td>53.170334</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Generate the Plot (with Error Bars)
# Creating a scatter plot that shows how the tumor volume changes over time for each treatment.
plt.errorbar(data_munging.index, data_munging["Capomulin"], yerr = data_munging_sem["Capomulin"],
             color = "r", marker = "o", linestyle = "--", linewidth = 0.5)

plt.errorbar(data_munging.index, data_munging["Infubinol"], yerr = data_munging_sem["Infubinol"],
             color = "b", marker = "^", linestyle = "--", linewidth = 0.5)

plt.errorbar(data_munging.index, data_munging["Ketapril"], yerr = data_munging_sem["Ketapril"],
             color = "g", marker = "s", linestyle = "--", linewidth = 0.5)

plt.errorbar(data_munging.index, data_munging["Placebo"], yerr = data_munging_sem["Placebo"],
             color = "black", marker = "d", linestyle = "--", linewidth = 0.5)


# capomulin = plt.errorbar(data_munging.index, data_munging["Capomulin"], yerr = data_munging_sem["Capomulin"],
#              color = "r", marker = 'o', linestyle = '--', linewidth = 1, label = "Capomulin")

# infubinol = plt.errorbar(data_munging.index, data_munging["Infubinol"], yerr = data_munging_sem["Infubinol"],
#              color = 'b', marker = '^', linestyle = '--', linewidth = 1, label = "Infubinol")

# ketapril = plt.errorbar(data_munging.index, data_munging["Ketapril"], yerr = data_munging_sem["Ketapril"],
#              color = 'g', marker = 's', linestyle = '--', linewidth = 1, label = "Ketapril")

# placebo = plt.errorbar(data_munging.index, data_munging["Placebo"], yerr = data_munging_sem["Placebo"],
#              color = 'black', marker = 'd', linestyle = '--', linewidth = 1, label = "Placebo")



# Chart title, xlabel, ylabel, legend, xlim, ylim
plt.title("Tumor Response to Treatment")
plt.xlabel("Time (Days)")
plt.ylabel("Tumor Volume (mm3)")

plt.legend(["Capomulin", "Infubinol", "Ketapril", "Placebo"], loc = "best")

# plt.legend()
ax = plt.axes()
ax.yaxis.grid(linestyle = "dotted")
# plt.grid()

# plt.set_xlim = (-5, 4)
# plt.set_ylim = (35, 75)

plt.show()
# plt.plot()
# plt.fig()

# Save the Figure
plt.savefig("TumorResponse.png")
```


![png](output_7_0.png)



    <Figure size 432x288 with 0 Axes>



```python
#Metastatic Response to Treatment
#Store the Mean Met. Site Data Grouped by Drug and Timepoint 

# metastatic_response = mouse_clinical.groupby(["Drug", "Timepoint"])["Metastatic Sites"]
metastatic_response = mouse_clinical.loc[:,["Timepoint", "Drug", "Metastatic Sites"]]

metastatic_means = metastatic_response.groupby(["Drug", "Timepoint"]).mean()

metastatic_means.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Metastatic Sites</th>
    </tr>
    <tr>
      <th>Drug</th>
      <th>Timepoint</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5" valign="top">Capomulin</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.160000</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.320000</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.375000</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.652174</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Store the Standard Error associated with Met. Sites Grouped by Drug and Timepoint 
metastatic_sem = mouse_clinical.groupby(["Drug", "Timepoint"])["Metastatic Sites"]
metastatic_sem_df = metastatic_sem.sem()

# Convert to DataFrame
metastatic_sem_df = metastatic_sem_df.reset_index()

# Preview DataFrame
metastatic_sem_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Drug</th>
      <th>Timepoint</th>
      <th>Metastatic Sites</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Capomulin</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Capomulin</td>
      <td>5</td>
      <td>0.074833</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Capomulin</td>
      <td>10</td>
      <td>0.125433</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Capomulin</td>
      <td>15</td>
      <td>0.132048</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Capomulin</td>
      <td>20</td>
      <td>0.161621</td>
    </tr>
  </tbody>
</table>
</div>




```python
 # Minor Data Munging to Re-Format the Data Frames
data_munging_meta = metastatic_means.pivot_table("Metastatic Sites", ["Timepoint"], "Drug")
data_munding_meta_sem = metastatic_sem_df.pivot_table("Metastatic Sites", ["Timepoint"], "Drug")

# Preview that Reformatting worked
data_munging_meta.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Ceftamin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Naftisol</th>
      <th>Placebo</th>
      <th>Propriva</th>
      <th>Ramicane</th>
      <th>Stelasyn</th>
      <th>Zoniferol</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.160000</td>
      <td>0.380952</td>
      <td>0.280000</td>
      <td>0.304348</td>
      <td>0.260870</td>
      <td>0.375000</td>
      <td>0.320000</td>
      <td>0.120000</td>
      <td>0.240000</td>
      <td>0.166667</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.320000</td>
      <td>0.600000</td>
      <td>0.666667</td>
      <td>0.590909</td>
      <td>0.523810</td>
      <td>0.833333</td>
      <td>0.565217</td>
      <td>0.250000</td>
      <td>0.478261</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.375000</td>
      <td>0.789474</td>
      <td>0.904762</td>
      <td>0.842105</td>
      <td>0.857143</td>
      <td>1.250000</td>
      <td>0.764706</td>
      <td>0.333333</td>
      <td>0.782609</td>
      <td>0.809524</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.652174</td>
      <td>1.111111</td>
      <td>1.050000</td>
      <td>1.210526</td>
      <td>1.150000</td>
      <td>1.526316</td>
      <td>1.000000</td>
      <td>0.347826</td>
      <td>0.952381</td>
      <td>1.294118</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Generate the Plot (with Error Bars)
# Creating a scatter plot that shows how the tumor volume changes over time for each treatment.
plt.errorbar(data_munging_meta.index, data_munging_meta["Capomulin"], yerr = data_munding_meta_sem["Capomulin"],
             color = "r", marker = "o", linestyle = "--", linewidth = 0.5)

plt.errorbar(data_munging_meta.index, data_munging_meta["Infubinol"], yerr = data_munding_meta_sem["Infubinol"],
             color = "b", marker = "^", linestyle = "--", linewidth = 0.5)

plt.errorbar(data_munging_meta.index, data_munging_meta["Ketapril"], yerr = data_munding_meta_sem["Ketapril"],
             color = "g", marker = "s", linestyle = "--", linewidth = 0.5)

plt.errorbar(data_munging_meta.index, data_munging_meta["Placebo"], yerr = data_munding_meta_sem["Placebo"],
             color = "black", marker = "d", linestyle = "--", linewidth = 0.5)

# Chart title, xlabel, ylabel, legend, xlim, ylim
plt.title("Metastatic Spread During Treatment")
plt.xlabel("Treatment Duration (Days)")
plt.ylabel("Met. Sites")
# plt.legend(loc = "upper left")

plt.legend(["Capomulin", "Infubinol", "Ketapril", "Placebo"], loc = "best")

# plt.grid(alpha = 0.5)
ax = plt.axes()
ax.yaxis.grid(linestyle = "dotted")

plt.set_xlim = (-5, 45)
plt.set_ylim = (-0.5, 4)

plt.show()

# Save the Figure
plt.savefig("MetastaticSpread.png")
```


![png](output_11_0.png)



    <Figure size 432x288 with 0 Axes>



```python
# Subset the data to be grouped by Drug and Timepoint and take a count of Mouse ID to find overal survival

# count_mouse = mouse_clinical.loc[:,["Timepoint", "Drug", "Mouse ID"]]
count_mouse = mouse_clinical.groupby(["Drug", "Timepoint"])

count_mouse_df = count_mouse[["Mouse ID"]].count().rename(columns={"Mouse ID": "Mouse Count"})

count_mouse_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Mouse Count</th>
    </tr>
    <tr>
      <th>Drug</th>
      <th>Timepoint</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5" valign="top">Capomulin</td>
      <td>0</td>
      <td>25</td>
    </tr>
    <tr>
      <td>5</td>
      <td>25</td>
    </tr>
    <tr>
      <td>10</td>
      <td>25</td>
    </tr>
    <tr>
      <td>15</td>
      <td>24</td>
    </tr>
    <tr>
      <td>20</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>




```python
 # Minor Data Munging to Re-Format the Data Frames
data_munging_mouse = count_mouse_df.pivot_table("Mouse Count", ["Timepoint"], "Drug")

# Preview that Reformatting worked
data_munging_mouse.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Drug</th>
      <th>Capomulin</th>
      <th>Ceftamin</th>
      <th>Infubinol</th>
      <th>Ketapril</th>
      <th>Naftisol</th>
      <th>Placebo</th>
      <th>Propriva</th>
      <th>Ramicane</th>
      <th>Stelasyn</th>
      <th>Zoniferol</th>
    </tr>
    <tr>
      <th>Timepoint</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>26</td>
      <td>25</td>
      <td>26</td>
      <td>25</td>
    </tr>
    <tr>
      <td>5</td>
      <td>25</td>
      <td>21</td>
      <td>25</td>
      <td>23</td>
      <td>23</td>
      <td>24</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>24</td>
    </tr>
    <tr>
      <td>10</td>
      <td>25</td>
      <td>20</td>
      <td>21</td>
      <td>22</td>
      <td>21</td>
      <td>24</td>
      <td>23</td>
      <td>24</td>
      <td>23</td>
      <td>22</td>
    </tr>
    <tr>
      <td>15</td>
      <td>24</td>
      <td>19</td>
      <td>21</td>
      <td>19</td>
      <td>21</td>
      <td>20</td>
      <td>17</td>
      <td>24</td>
      <td>23</td>
      <td>21</td>
    </tr>
    <tr>
      <td>20</td>
      <td>23</td>
      <td>18</td>
      <td>20</td>
      <td>19</td>
      <td>20</td>
      <td>19</td>
      <td>17</td>
      <td>23</td>
      <td>21</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Generate the Plot (Accounting for percentages)
plt.plot(np.arange(0, 50, 5), (count_mouse_df.loc["Capomulin", "Mouse Count"]/25) * 100,
         color = "r", marker = "o", linestyle = "--", linewidth = 0.5, label = "Capomulin")
plt.plot(np.arange(0, 50, 5), (count_mouse_df.loc["Infubinol", "Mouse Count"]/25) * 100,
         color = "b", marker = "^", linestyle = "--", linewidth = 0.5, label = "Infubinol")
plt.plot(np.arange(0, 50, 5), (count_mouse_df.loc["Ketapril", "Mouse Count"]/25) * 100,
         color = "g", marker = "s", linestyle = "--", linewidth = 0.5, label = "Ketapril")
plt.plot(np.arange(0, 50, 5), (count_mouse_df.loc["Placebo", "Mouse Count"]/25) * 100,
         color = "black", marker = "d", linestyle = "--", linewidth = 0.5, label = "Placebo")

# Add gridlines
plt.grid(alpha = 0.5)

# Chart title, xlabel, ylabel, legend, xlim, ylim
plt.title("Survival During Treatment")
plt.xlabel("Time (Days)")
plt.ylabel("Survival Rate (%)")
plt.legend(loc = "lower left")

# Add x limits and y limits
plt.xlim(-2.5,47)
plt.ylim(33,103)

# Plot the graph
plt.show()

# Save the Figure
plt.savefig("SurvivalDuring.png")
```


![png](output_14_0.png)



    <Figure size 432x288 with 0 Axes>



```python
percent_change = ((data_munging.loc[45, :] - data_munging.loc[0, :]) / data_munging.loc[0, :]) * 100
percent_change
```




    Drug
    Capomulin   -19.475303
    Ceftamin     42.516492
    Infubinol    46.123472
    Ketapril     57.028795
    Naftisol     53.923347
    Placebo      51.297960
    Propriva     47.241175
    Ramicane    -22.320900
    Stelasyn     52.085134
    Zoniferol    46.579751
    dtype: float64




```python
 # Store all Relevant Percent Changes into a Tuple
drugs = ["Capomulin", "Infubinol", "Ketapril", "Placebo"]

fig, ax = plt.subplots()
x_axis = np.arange(0, 4)
percent_drugs = [percent_change["Capomulin"], percent_change["Infubinol"], percent_change["Ketapril"], percent_change["Placebo"]]

colors = []

# Splice the data between passing and failing drugs
for percent in percent_drugs:
    if percent >= 0:
        colors.append("r")
    else:
        colors.append("g")

barplot = ax.bar(x_axis, percent_drugs, width = 1, align = "center", color = colors,
                 linewidth = 1, tick_label = drugs)

# Orient widths. Add labels, tick marks, etc.
ax.set_title("Tumor Change over 45 Day Treatment")
ax.set_ylabel("% Tumor Volume Change")
ax.grid(alpha = 0.5)
ax.set_xlim(-0.8, 3.8)
ax.set_ylim(-30, 70)

for p in barplot:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate("{:,.2%}".format(height), (x, y))


# Add labels for the percentages
autolabel(barplot, ax)

plt.tight_layout()
fig.show()

# Save the Figure
plt.savefig("TumorChange.png")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-15-e1ce4696902a> in <module>
         32 
         33 # Add labels for the percentages
    ---> 34 autolabel(barplot, ax)
         35 
         36 plt.tight_layout()
    

    NameError: name 'autolabel' is not defined



![png](output_16_1.png)





