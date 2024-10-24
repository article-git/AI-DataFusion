# %% [markdown]
# Data fusion module for TATP of IR, RAMAN, IMS and GC-QEPAS sensors

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager
import json
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA

# %%
from sklearn import svm
import plotly.graph_objects as go
from scipy.signal import savgol_filter
import random

# %%
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
from scipy.spatial.distance import cdist


# %% [markdown]
# IMS for identification

# %% [markdown]
# Import IMS training data

# %%
data_train_IMS = pd.read_excel('/Users/giorgiofelizzato/Desktop/RISEN project/data fusion/Moduli python data fusion-final version/TATP data for DF/IMS data/TATP_IMS.xlsx', sheet_name='IMS', index_col=0, header=0)
data_train_IMS.head()

# %%
data_train_IMS.shape

# %%
y_train_IMS = data_train_IMS.loc[:,'Class']
y_train_IMS

# %%
X_train_IMS = data_train_IMS.iloc[:,1:]
X_train_IMS

# %%
#training of the one class support vector machine model for IMs data
clf_IMS = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma="auto", degree=3)
clf_IMS.fit(X_train_IMS, y_train_IMS)

# %%
y_pred_train_IMS = clf_IMS.predict(X_train_IMS)
y_pred_train_IMS

# %%
clf_IMS.score_samples(X_train_IMS)

# %%
#Signed distance to the separating hyperplane. 
#Signed distance is positive for an inlier and negative for an outlier.
clf_IMS.decision_function(X_train_IMS)

# %%
# Plotting the training set data for score samples
Z_train = clf_IMS.score_samples(X_train_IMS)

plt.title("Training set Detection")
plt.plot(Z_train, marker='o', linestyle='-', color='b', label='Training Samples')

plt.xlabel("Sample Index")
plt.ylabel("Score Samples Value")
plt.axhline(y=0, color='r', linestyle='--', label='Decision Boundary')

plt.legend()
plt.show()

# %%
# Plotting the training set data
Z_train = clf_IMS.decision_function(X_train_IMS)

plt.title("Training set Detection")
plt.plot(Z_train, marker='o', linestyle='-', color='b', label='Training Samples')

plt.xlabel("Sample Index")
plt.ylabel("Decision Function Value")
plt.axhline(y=-0.2, color='r', linestyle='--', label='Decision Boundary')

plt.legend()
plt.show()

# %% [markdown]
# import new data of IMS as json file (new data for the prediction)

# %%
file_path_IMS = '/Users/giorgiofelizzato/Desktop/RISEN project/data fusion/Moduli python data fusion-final version/TATP data for DF/measurement_IMS_TATP_txt.json'

# Open the file and load its content
with open(file_path_IMS, 'r') as file:
    data_IMS = json.load(file)

# Print the data to verify
print(data_IMS)

# %%
# print data on column
for key, value in data_IMS.items():
    print(f"{key}: {value}")

# %%
# Function to navigate into the json file levels
def find_data(json_data, target_key, current_level=1, max_level=5):
    if current_level > max_level:
        return None
    
    for key, value in json_data.items():
        if key == target_key:
            return value
        elif isinstance(value, dict):
            result = find_data(value, target_key, current_level=current_level + 1, max_level=max_level)
            if result is not None:
                return result
    return None

# Aopen the json file
with open(file_path_IMS, 'r') as file:
    data_IMS = json.load(file)

# key in the level 
target_key_IMS = 'columns'  

result_IMS = find_data(data_IMS, target_key_IMS)

if result_IMS is not None:
    print(f"{result_IMS}")
else:
    print(f"{target_key_IMS} not found")

# %%
def find_data(json_data, target_key, current_level=1, max_level=5):
    if current_level > max_level:
        return None
    
    for key, value in json_data.items():
        if key == target_key:
            return value
        elif isinstance(value, dict):
            result = find_data(value, target_key, current_level=current_level + 1, max_level=max_level)
            if result is not None:
                return result
    return None

# open json file
with open(file_path_IMS, 'r') as file:
    data_IMS = json.load(file)

#  target
target_key_IMS = 'columns'

result_IMS = find_data(data_IMS, target_key_IMS)

if result_IMS is not None and len(result_IMS) >= 2:
    x = result_IMS[0][1:]  
    y = result_IMS[1][1:]  
    print(f"x: {x}")
    print(f"y: {y}")
else:
    print(f"{target_key_IMS} not found.")

# %%
def find_data(json_data, target_key, current_level=1, max_level=5):
    if current_level > max_level:
        return None
    
    for key, value in json_data.items():
        if key == target_key:
            return value
        elif isinstance(value, dict):
            result = find_data(value, target_key, current_level=current_level + 1, max_level=max_level)
            if result is not None:
                return result
    return None

# open json file
with open(file_path_IMS, 'r') as file:
    data_IMS = json.load(file)

#  target
target_key_IMS = 'columns'

result_IMS = find_data(data_IMS, target_key_IMS)

if result_IMS is not None and len(result_IMS) >= 2:
    y = result_IMS[1][1:]  
    IMS_plasmagram = np.array(y)  
    print(f"{IMS_plasmagram}")
else:
    print(f"{target_key_IMS} not found.")

# %%
def generate_random_numbers(count):
    return [round(random.uniform(0, 0.007), 2) for _ in range(count)]

total_values_needed = 668

IMS_plasmagram_list = IMS_plasmagram.tolist()

num_random_numbers = total_values_needed - len(IMS_plasmagram_list)

if num_random_numbers > 0:
    random_numbers = generate_random_numbers(num_random_numbers)
    final_values = random_numbers + IMS_plasmagram_list[:total_values_needed]
else:
    final_values = IMS_plasmagram_list[:total_values_needed]  #
final_values = [round(val, 4) for val in final_values]

IMS_plasmagram_final_array = np.array(final_values)

print("Array plasmagram:")
print(IMS_plasmagram_final_array)



# %%
def find_data(json_data, target_key, current_level=1, max_level=7):
    if current_level > max_level:
        return None
    
    for key, value in json_data.items():
        if key == target_key:
            return value
        elif isinstance(value, dict):
            result = find_data(value, target_key, current_level=current_level + 1, max_level=max_level)
            if result is not None:
                return result
    return None


with open(file_path_IMS, 'r') as file:
    data_IMS = json.load(file)


target_key_IMS = 'columns'  

result_IMS = find_data(data_IMS, target_key_IMS)

if result_IMS is not None and len(result_IMS) >= 2:
    x = result_IMS[0][1:]  
    y = result_IMS[1][1:]  
    
   
    df_x = pd.DataFrame(x, columns=['x'])
    
    #
    df_y = pd.DataFrame(y, columns=['y'])
    
    
    df_combined_IMS = pd.concat([df_x, df_y], axis=1)
    
    print("DataFrame plasmagram:")
    print(df_combined_IMS)
else:
    print(f"{target_key_IMS} not found")

# %%
def find_data(json_data, target_key, current_level=1, max_level=5):
    if current_level > max_level:
        return None
    
    for key, value in json_data.items():
        if key == target_key:
            return value
        elif isinstance(value, dict):
            result = find_data(value, target_key, current_level=current_level + 1, max_level=max_level)
            if result is not None:
                return result
    return None

with open(file_path_IMS, 'r') as file:
    data = json.load(file)

target_key_IMS = 'columns' 

result_IMS = find_data(data, target_key_IMS)

if result_IMS is not None and len(result_IMS) >= 2:
    x = result_IMS[0][1:]  
    y = result_IMS[1][1:]  
    
    df_x = pd.DataFrame(x, columns=['x'])
    
    df_measurement_y = pd.DataFrame(y, columns=['y'])
    
    df_combined_IMS = pd.concat([df_x, df_measurement_y], axis=1)
    
    df_combined_IMS['x'] = pd.to_numeric(df_combined_IMS['x'])
    df_combined_IMS['y'] = pd.to_numeric(df_combined_IMS['y'])
    
    target_value = 2.06
    tolerance = 0.02 * target_value
    lower_bound = target_value - tolerance
    upper_bound = target_value + tolerance
    
    filtered_df = df_combined_IMS[(df_combined_IMS['x'] >= lower_bound) & (df_combined_IMS['x'] <= upper_bound) & (df_combined_IMS['y'] > 1)]
    
    if not filtered_df.empty:
        print("TATP identified by means of Reduced Mobility (IMS sensor):")
        print(filtered_df)
    else:
        print("TATP NOT identified by means of Reduced Mobility (IMS sensor)")
else:
    print(f"key data '{target_key_IMS}' not find.")

# %%
# Convert IMS_plasmagram to a single row DataFrame with the appropriate column names
processed_dataframe_ims = pd.DataFrame([IMS_plasmagram_final_array], columns=data_train_IMS.columns[1:])

# Display the DataFrame
print(processed_dataframe_ims.head())

# %%
y_pred_unknown_IMS = clf_IMS.predict(processed_dataframe_ims)
y_pred_unknown_IMS

# %%
clf_IMS.score_samples(processed_dataframe_ims)

# %%
# Use the decision function to get the signed distance
decision_values_IMS_unknown = clf_IMS.decision_function(processed_dataframe_ims)
decision_values_IMS_unknown

# %%
lower_threshold = -0.50  
upper_threshold = 10     

predictions = np.where(
    (decision_values_IMS_unknown >= lower_threshold) & (decision_values_IMS_unknown <= upper_threshold),
    "TATP identified by means of OC-SVM (IMS sensor)",
    "TATP NOT identified by means of OC-SVM (IMS sensor)"
)

print(predictions)

# %%
Z_train = clf_IMS.score_samples(X_train_IMS)
decision_function_scores_unknown = clf_IMS.score_samples(processed_dataframe_ims)

# Convert the scores to a pandas DataFrame
scores_df_train = pd.DataFrame({'Sample Index': X_train_IMS.index, 'Decision Function Scores': Z_train})
scores_df_unknown = pd.DataFrame({'Sample Index': [X_train_IMS.index.max() + 1], 'Decision Function Scores': decision_function_scores_unknown})
scores_df_unknown['Decision Function Scores'] = scores_df_unknown['Decision Function Scores'].round(decimals=2)

# Define thresholds
lower_threshold = -0.5
upper_threshold = 6

# Predict inliers and outliers based on the threshold for score samples
unknown_prediction = "Acetone"
if scores_df_unknown['Decision Function Scores'].iloc[0] < lower_threshold or scores_df_unknown['Decision Function Scores'].iloc[0] > upper_threshold:
    unknown_prediction = "Unknown sample"

# Create traces for training samples, unknown sample, and decision boundary
trace_train = go.Scatter(x=scores_df_train['Sample Index'], y=scores_df_train['Decision Function Scores'], mode='lines+markers',
                         name='Training Samples', line=dict(color='blue'))

trace_unknown = go.Scatter(x=scores_df_unknown['Sample Index'], y=scores_df_unknown['Decision Function Scores'], mode='markers',
                           name=f'Unknown Sample ({unknown_prediction})', marker=dict(color='red', size=10))

# Add labels for training samples
predictions = np.where(scores_df_train['Decision Function Scores'] < lower_threshold, "Unknown sample",
                       np.where(scores_df_train['Decision Function Scores'] > upper_threshold, "Unknown sample", "Acetone"))
annotations_train = [dict(x=scores_df_train['Sample Index'][i], y=scores_df_train['Decision Function Scores'][i],
                          text=predictions[i], showarrow=True, arrowhead=2, ax=0, ay=-30) for i in range(len(predictions))]

# Add label for unknown sample
annotations_unknown = [dict(x=scores_df_unknown['Sample Index'].iloc[0], y=scores_df_unknown['Decision Function Scores'].iloc[0],
                            text=unknown_prediction, showarrow=True, arrowhead=2, ax=0, ay=-30)]

# Create decision boundary traces
lower_decision_boundary_trace = go.Scatter(x=[scores_df_train['Sample Index'].min(), scores_df_train['Sample Index'].max()],
                                            y=[lower_threshold, lower_threshold], mode='lines', name='Lower Threshold',
                                            line=dict(color='green', dash='dash'))
upper_decision_boundary_trace = go.Scatter(x=[scores_df_train['Sample Index'].min(), scores_df_train['Sample Index'].max()],
                                            y=[upper_threshold, upper_threshold], mode='lines', name='Upper Threshold',
                                            line=dict(color='orange', dash='dash'))

# Create layout
layout = go.Layout(title="One-Class SVM Detection of Acetone", xaxis=dict(title="Sample Index"), yaxis=dict(title="Decision Function Value"),
                   legend=dict(x=1, y=0.5), annotations=annotations_train + annotations_unknown)

# Create figure
fig = go.Figure(data=[trace_train, trace_unknown, lower_decision_boundary_trace, upper_decision_boundary_trace], layout=layout)

fig.show()

# %%
# OneClassSVM model
Z_train = clf_IMS.decision_function(X_train_IMS)
decision_function_scores_unknown = clf_IMS.decision_function(processed_dataframe_ims)

# Convert the scores to a pandas DataFrame
scores_df_train = pd.DataFrame({'Sample Index': X_train_IMS.index, 'Decision Function Scores': Z_train})
scores_df_unknown = pd.DataFrame({'Sample Index': [X_train_IMS.index.max() + 1], 'Decision Function Scores': decision_function_scores_unknown})
scores_df_unknown['Decision Function Scores'] = scores_df_unknown['Decision Function Scores'].round(decimals=2)

# Predict inliers and outliers based on the threshold for decision function value
threshold = -0.5
unknown_prediction = "unknown" if scores_df_unknown['Decision Function Scores'].iloc[0] < threshold else "TATP"

# Create traces for training samples, unknown sample, and decision boundary
trace_train = go.Scatter(x=scores_df_train['Sample Index'], y=scores_df_train['Decision Function Scores'], mode='lines+markers',
                         name='Training Samples', line=dict(color='blue'))

trace_unknown = go.Scatter(x=scores_df_unknown['Sample Index'], y=scores_df_unknown['Decision Function Scores'], mode='markers',
                           name=f'Unknown Sample ({unknown_prediction})', marker=dict(color='red', size=10))

# Add labels for training samples
predictions = np.where(scores_df_train['Decision Function Scores'] < threshold, "unknown", "TATP")
annotations_train = [dict(x=scores_df_train['Sample Index'][i], y=scores_df_train['Decision Function Scores'][i],
                           text=predictions[i], showarrow=True, arrowhead=2, ax=0, ay=-30) for i in range(len(predictions))]

# Add label for unknown sample
annotations_unknown = [dict(x=scores_df_unknown['Sample Index'].iloc[0], y=scores_df_unknown['Decision Function Scores'].iloc[0],
                             text=unknown_prediction, showarrow=True, arrowhead=2, ax=0, ay=-30)]

# Create decision boundary trace
decision_boundary_trace = go.Scatter(x=[scores_df_train['Sample Index'].min(), scores_df_train['Sample Index'].max()],
                                     y=[threshold, threshold], mode='lines', name='Decision Boundary', line=dict(color='red', dash='dash'))

# Create layout
layout = go.Layout(title="One-Class SVM Detection of TATP", xaxis=dict(title="Sample Index"), yaxis=dict(title="Decision Function Scores"),
                   legend=dict(x=1, y=0.5), annotations=annotations_train + annotations_unknown)

# Create figure
fig = go.Figure(data=[trace_train, trace_unknown, decision_boundary_trace], layout=layout)

fig.show()

# %% [markdown]
# IR for identification of TATP

# %% [markdown]
# import the IR data of unknown sample (for detection)

# %%
# Path JSON file
file_path_IR = '/Users/giorgiofelizzato/Desktop/RISEN project/data fusion/Moduli python data fusion-final version/TATP data for DF/measurement_TATP_IR.json'

# Open the file and load its content
with open(file_path_IR, 'r') as file:
    data_IR = json.load(file)

# Print the data to verify
print(data_IR)

# %%
# print data on column
for key, value in data_IR.items():
    print(f"{key}: {value}")

# %%
# Function to navigate into the json file levels
def find_data(json_data, target_key, current_level=1, max_level=5):
    if current_level > max_level:
        return None
    
    for key, value in json_data.items():
        if key == target_key:
            return value
        elif isinstance(value, dict):
            result = find_data(value, target_key, current_level=current_level + 1, max_level=max_level)
            if result is not None:
                return result
    return None

# Aopen the json file
with open(file_path_IR, 'r') as file:
    data_IR = json.load(file)

# key in the level 
target_key_IR = 'columns'  

result_IR = find_data(data_IR, target_key_IR)

if result_IR is not None:
    print(f"{result_IR}")
else:
    print(f"{target_key_IR} not found")

# %%
print(data_IR.keys())  

# %%
with open(file_path_IR, 'r') as file:
    data_IR = json.load(file)


x_values_IR = data_IR['data']['columns'][0][1:]  

reflectance_values = {}

for spectrum in data_IR['data']['columns'][1:]:
    spectrum_name = spectrum[0]  
    spectrum_values = spectrum[1:]  
    reflectance_values[spectrum_name] = spectrum_values

print("Wavelenghts (x):", x_values_IR)
print("\nSignal:")
for spectrum_name, spectrum_values in reflectance_values.items():
    print(f"{spectrum_name}: {spectrum_values}")

# %%
import pandas as pd

# Supponiamo che tu abbia già eseguito il codice precedente e che tu abbia i dati 'x_values' e 'reflectance_values'

# Converti i dati in un DataFrame
df = pd.DataFrame(reflectance_values, index=x_values_IR)

# Imposta il nome degli indici e delle colonne (opzionale)
df.index.name = 'Wavelenghts'
df.columns.name = 'Spectrum'

# Mostra le prime righe del DataFrame
print(df.head())

# %%
x_values_IR

# %%
spectra_arrays_IR = {spectrum_name: np.array(spectrum_values) for spectrum_name, spectrum_values in reflectance_values.items()}

for spectrum_name, spectrum_array in spectra_arrays_IR.items():
    print(f"Spectrum: {spectrum_name}")
    print("Array of intensity:", spectrum_array[:])  

# %%
spectra_arrays_numbered_IR = {i: np.array(spectrum_values) for i, (spectrum_name, spectrum_values) in enumerate(reflectance_values.items())}

for index, spectrum_array in spectra_arrays_numbered_IR.items():
    print(f"Index: {index}")
    print(spectrum_array[:])  

# %%
spectra_arrays_numbered_IR = {i: np.array(spectrum_values) for i, (spectrum_name, spectrum_values) in enumerate(reflectance_values.items())}

desired_count = 4
array_length = 967

for i in range(len(spectra_arrays_numbered_IR), desired_count):
    spectra_arrays_numbered_IR[i] = np.zeros(array_length)

for index in range(desired_count):
    spectrum_array = spectra_arrays_numbered_IR[index]
    print(f"Index: {index}")
    print("Array of intensity:", spectrum_array)  

# %%
Index = 0
if Index in spectra_arrays_numbered_IR:
    array_0 = spectra_arrays_numbered_IR[Index]
    print(f"Array index {Index}:")
    print(array_0)  
else:
    print(f"index {Index} NOT find.")

# %%
Index = 1
if Index in spectra_arrays_numbered_IR:
    array_1 = spectra_arrays_numbered_IR[Index]
    print(f"Array index {Index}:")
    print(array_1)  
else:
    print(f"index {Index} NOT find.")

# %%
Index = 2
if Index in spectra_arrays_numbered_IR:
    array_2 = spectra_arrays_numbered_IR[Index]
    print(f"Array index {Index}:")
    print(array_2)  
else:
    print(f"index {Index} NOT find.")

# %%
# Aggiungi una virgola e uno spazio dopo ogni numero
array_con_virgole_e_spazi = [f"{num}, " for num in array_2]

# Unisci le stringhe in una sola stringa
risultato_finale = ''.join(array_con_virgole_e_spazi)

# Rimuovi la virgola e lo spazio finali se necessario
risultato_finale = risultato_finale.rstrip(', ')

# Mostra il risultato
print(risultato_finale)

# %%
Index = 3
if Index in spectra_arrays_numbered_IR:
    array_3 = spectra_arrays_numbered_IR[Index]
    print(f"Array index {Index}:")
    print(array_3)  
else:
    print(f"index {Index} NOT find.")

# %%
array_3.shape

# %%
Index = 4
if Index in spectra_arrays_numbered_IR:
    array_4 = spectra_arrays_numbered_IR[Index]
    print(f"Array index {Index}:")
    print(array_4)  
else:
    print(f"index {Index} NOT find.")

# %% [markdown]
# Predictive models for IR (FTIR)

# %%
reference_spectrum_TATP_FTIR = [ 1.0761509840956673, 1.1050045940712117, 1.1338582971754305, 1.1627121824162605, 1.190599410643601, 1.2182262336138254, 1.2591141933236307, 1.2923709047006733, 1.311658551332995, 1.3334406453328376, 1.3567179659419368, 1.392403963085369, 1.4249259653462678, 1.4520203015891602, 1.476744317077716, 1.4998975167484623, 1.5105771767000185, 1.5197603956640897, 1.526535279913923, 1.5532746676436358, 1.5869224441930103, 1.6080768566178993, 1.617520264323671, 1.6092650859501811, 1.6096937499062867, 1.6138692743075504, 1.6259607026522187, 1.6349250988287478, 1.6394482029403838, 1.6365460921656125, 1.631877485299591, 1.6341892641264035, 1.6425181822832882, 1.658883978846072, 1.6568274727178032, 1.6494208304911535, 1.6442601844527067, 1.6403629556350192, 1.6380535349461236, 1.6231720118257655, 1.603540587748208, 1.574842455081985, 1.5514552361270315, 1.5343508628043654, 1.5057235579379638, 1.474392639923158, 1.444616533397674, 1.415951072354208, 1.388522786024889, 1.376701777633275, 1.3671257471955547, 1.349692673686404, 1.34723862078732, 1.3604987551540062, 1.3575898778805195, 1.3502516872898473, 1.337681092462326, 1.3317829283280984, 1.3324777611884342, 1.3286382062871132, 1.3210795485377618, 1.3004425855922979, 1.2949219000172625, 1.303468963902567, 1.321656039291714, 1.3408209638743422, 1.3573370461392558, 1.3838865709572084, 1.4192294499227684, 1.4546672132114915, 1.4847226107413423, 1.4944259221692682, 1.5104882921874292, 1.5317977542663859, 1.5607461824387898, 1.5882814606821345, 1.607550052869029, 1.6246166164267954, 1.6399730939215158, 1.640090277349115, 1.637888940810634, 1.6336309446950747, 1.628086674203079, 1.621602690306535, 1.606987573287882, 1.5978828341048896, 1.6070998169110329, 1.626979419175431, 1.6541828408656987, 1.6918220743590047, 1.727358539635346, 1.7551017845038719, 1.7843266387531471, 1.8145076054599505, 1.8276239072868228, 1.8413353669446109, 1.858989561057279, 1.8749807493453245, 1.8899645398742206, 1.8994602916639856, 1.9057773878223776, 1.9051765112315415, 1.8990791751405272, 1.8898594371738295, 1.9043457454481665, 1.919868987882473, 1.9363849795445145, 1.9462047005037835, 1.9524612682450706, 1.954064254936371, 1.956730091094363, 1.9616461639270004, 1.9636680104397461, 1.9643034324942013, 1.9739359557747276, 1.9762810399658242, 1.9647483053692263, 1.9644050318715514, 1.968936573088391, 1.9581783807492152, 1.9505673342855259, 1.9485686979664616, 1.951515015250186, 1.9564476576210321, 1.9571622187746136, 1.9569189968411462, 1.9550743615861326, 1.9522521094737204, 1.9489333024400306, 1.9437083544309879, 1.9442094386200104, 1.9536959887198495, 1.9587035409658524, 1.961172376955338, 1.951425050461012, 1.9366026486013033, 1.9142992097215237, 1.905968797636345, 1.9022646563129246, 1.8960679667944356, 1.8901028347193383, 1.8844585751015068, 1.8786335170412753, 1.8735567916412694, 1.8760887367866335, 1.8693512693305185, 1.8505316020007587, 1.8460106264974405, 1.8457708065626866, 1.8460859577875932, 1.8431911254187665, 1.8363587131749577, 1.8348753818950263, 1.8350640961926639, 1.8368379332163498, 1.8348617632690027, 1.8285549401099486, 1.834898623103801, 1.8437914709821421, 1.8492265289833012, 1.8440041986317417, 1.8271919584465932, 1.8346722909685895, 1.8447701426970575, 1.8396264825856257, 1.8353564869747119, 1.8319813356097856, 1.827542801702434, 1.8239721618224254, 1.8256601510896926, 1.8249283217674637, 1.8218620792177218, 1.823328111396172, 1.8252708777363431, 1.8257218723794426, 1.8239314246616827, 1.8201044664560546, 1.812388427642717, 1.8042805929305126, 1.7970868414783057, 1.789496415989153, 1.7815666138239534, 1.763533610527042, 1.7436602007922621, 1.7222342666901476, 1.7040945212623533, 1.6886014899560262, 1.6804000066092077, 1.670198288288696, 1.6503804539821203, 1.6288558883724502, 1.6060378000909752, 1.5922986372898513, 1.579649993051112, 1.5674981317752756, 1.5539873753027116, 1.5395079704591383, 1.5301124412581606, 1.5191562188567656, 1.502550647847672, 1.489756545025977, 1.479516125889845, 1.4815099154527114, 1.477070126890371, 1.4535481026953185, 1.440132310754795, 1.4330760503305433, 1.4306505950968973, 1.4251755200970313, 1.4116800321883938, 1.4038297575307845, 1.3993128468880625, 1.3856903638491684, 1.3699646439503803, 1.350160409616535, 1.3388185130207673, 1.3321604219637568, 1.3275491988076311, 1.3203574885780558, 1.307611852555737, 1.3041356711664276, 1.3054627526324032, 1.2958930115119645, 1.2861063442870118, 1.275954560572684, 1.262542190242097, 1.247528805936981, 1.2307465471624999, 1.214675712230896, 1.1999243557087351, 1.1961604363851541, 1.1970502126019422, 1.1865817107174967, 1.17327587957385, 1.155039607422405, 1.153766546320806, 1.1589328982624671, 1.1479440328161663, 1.1336578196551692, 1.1139971893529113, 1.1012253594044816, 1.091289711069455, 1.083643087185285, 1.071363615704515, 1.0519944835077244, 1.0419375338253798, 1.0358910682920701, 1.0379522822733946, 1.0351326999045878, 1.0252947461417992, 1.0190198089221851, 1.013359449945275, 1.001571795405556, 0.984476371845372, 0.9602042465110204, 0.9478916688547959, 0.9389500777579807, 0.927060296560089, 0.9185211664259063, 0.9142444291414838, 0.8977419042925509, 0.876977964894899, 0.8503006573163733, 0.8315361691382025, 0.8222462410315093, 0.8117006054396562, 0.7976142740226505, 0.7623464874268558, 0.7309260048499365, 0.7038429287146528, 0.7003451177522294, 0.6993043562381065, 0.6800379498482823, 0.6658196739798462, 0.6569611416279563, 0.637686978841082, 0.6210465923834205, 0.62964977441969, 0.6399428655681201, 0.6519258658287185, 0.6530428575708143, 0.6521203845914939, 0.6514595001685077, 0.6500197234465198, 0.6478463388615194, 0.6399361010209835, 0.6333857603891442, 0.6367335918395352, 0.6365785626782141, 0.6333164763641307, 0.6418299142322996, 0.6501320696760395, 0.6507610783790128, 0.6470044924852529, 0.6395852122030111, 0.6365871576307047, 0.6353475351983016, 0.6382325355983334, 0.6372275813542871, 0.6331647483258609, 0.6313089165317556, 0.628153069843432, 0.6199936012095688, 0.6160820343671198, 0.6153121448912342, 0.6103068053397652, 0.6052826707735262, 0.6014011549883922, 0.5994123167392389, 0.5987397060151088, 0.5995153162288673, 0.6007065948247069, 0.6027334832249798, 0.5985997604848473, 0.5904402918509825, 0.5892990248367082, 0.5877538314002991, 0.5840965191898753, 0.5872826449286807, 0.5946674016798144, 0.5861184819945435, 0.576842712826977, 0.5674963222707406, 0.5595983266569559, 0.5525338413402376, 0.546890685817336, 0.540994811641773, 0.5344529945088315, 0.5399745627024757, 0.5520003524777402, 0.5557210614150795, 0.554606251185782, 0.5437176648828356, 0.5406945729289443, 0.5416343644550545, 0.540811304180576, 0.539816212720976, 0.5384891312550004, 0.5380639883901522, 0.5381315548092369, 0.541892752138255, 0.5490959096950305, 0.5625177348901822, 0.5673446946250931, 0.5681320149042868, 0.5612594665313028, 0.5634186662181577, 0.5808735682644217, 0.5971152329307214, 0.6125835731127119, 0.6225511609574873, 0.6282415017565637, 0.6271346670178951, 0.624684110070651, 0.6221574107658008, 0.6261097247822637, 0.6267094296000992, 0.6223052400916242, 0.6151239429664715, 0.6065670715329008, 0.5933025053951217, 0.5763285421573435, 0.554150748455532, 0.5429602297640108, 0.5355482786917758, 0.5301238435222672, 0.5262263415059526, 0.5243435429556585, 0.512556463019311, 0.4988244640023156, 0.493556588199056, 0.48563146236144517, 0.47440706634694907, 0.46340901531889434, 0.45248394499243705, 0.44163590526599045, 0.43106529717891134, 0.42081895983915346, 0.41648432242126043, 0.4133227108911263, 0.408217510122477, 0.4020995866748091, 0.3948670313477134, 0.3956137652889124, 0.39989007687757067, 0.41365947954820903, 0.42017036048935646, 0.41915716402797615, 0.412227404814831, 0.40528664273726966, 0.4043352158058641, 0.40332933002783095, 0.4022702743699536, 0.3875656137188926, 0.3700718704222664, 0.3513569140069505, 0.3399922083091706, 0.3353863538458979, 0.3137206865823894, 0.292888845561132, 0.2867872326721161, 0.28081812441469695, 0.274963754804754, 0.25622179750224283, 0.23886887794752465, 0.23352776798816818, 0.23621764898088618, 0.24545453346714433, 0.24448015127108363, 0.24300198480333507, 0.2441320842467592, 0.2375076425892281, 0.2249339498225024, 0.21882123639250367, 0.2154837544284895, 0.21867624739210675, 0.22558181229754187, 0.23516691365583994, 0.23667367405586506, 0.23758126622683987, 0.2388098302472787, 0.23327479663630762, 0.2231513927728743, 0.22821377274408075, 0.23419062782598282, 0.23975006407907756, 0.23846409408804178, 0.23281585541202418, 0.22623138691818742, 0.2219819764217765, 0.22364603764976582, 0.2303296818418469, 0.2400154689550349, 0.24604733887971428, 0.25463456888912406, 0.26943097416991557, 0.27745624792400025, 0.28168462549488427, 0.2794633109602262, 0.2800526844290753, 0.2869846625771125, 0.29855652206667527, 0.3125648489273651, 0.3122050945776154, 0.31127073567912134, 0.30935588024988153, 0.30605300473026115, 0.30198284404104947, 0.288460629596421, 0.2711838342442221, 0.2468510161519126, 0.22845261761656444, 0.21309687822670256, 0.21084784976099935, 0.2089313844191208, 0.2076003401133985, 0.20630796165191886, 0.20521405724411992, 0.2086129228783663, 0.21258199090160787, 0.21749234540827225, 0.22371376976886417, 0.23055622022659497, 0.23915228712914657, 0.24551182886484382, 0.24840561022747001, 0.24724640234495662, 0.24593484996179468, 0.2627777271731742, 0.27970154029018934, 0.29674318656393234, 0.3072613409910676, 0.31550104954434444, 0.3231773785821216, 0.3289375343045753, 0.33207484286492456, 0.3419984079224576, 0.3546413122172843, 0.37259217470770084, 0.38978259073022636, 0.40599380171189076, 0.42236492286455485, 0.4382840493382307, 0.45032167552075286, 0.46481877090012175, 0.4822965474915763, 0.500511168745014, 0.5178918676436076, 0.5283068532975026, 0.5416000537501006, 0.5581773710884015, 0.5689944985130506, 0.5788867992849612, 0.5915618848147035, 0.6042060142906932, 0.6168168804294255, 0.6187373483035499, 0.6170508943283823, 0.6086524705058726, 0.5994391579740741, 0.5894011387967331, 0.5632323855461894, 0.5366444597375005, 0.5231937592139395, 0.5001881001314109, 0.4680744980950253, 0.4409268754631531, 0.41545557205435607, 0.39343425812423777, 0.3745346461394335, 0.3584374711282498, 0.34385708178942676, 0.3306336940196628, 0.32179470550301503, 0.31603416781248156, 0.312875856510878, 0.3167544367070929, 0.32416441085477304, 0.3407772941551329, 0.35764294107405586, 0.3747097117324921, 0.3840150009900866, 0.3972698777221077, 0.4264693602766928, 0.4565659013858301, 0.48733406225047804, 0.5044750318290464, 0.5206264634127868, 0.537852207109201, 0.5549081150916416, 0.5718444448673112, 0.5923091110568321, 0.6128550835238936, 0.6328303420470149, 0.6529529961826572, 0.6731731807484703, 0.7063399961127996, 0.7385587509707409, 0.7661837693993141, 0.7962644726215664, 0.8278710382398728, 0.8630219053289149, 0.9000019776526768, 0.9409857304730722, 0.9811124237301264, 1.020739504066394, 1.0464255116303978, 1.0731108584387146, 1.1030024387512891, 1.123157390249071, 1.1379932574869058, 1.1563942332720383, 1.174222352822915, 1.1907383389419663, 1.207700340932855, 1.2248903962880706, 0.2513017660547729, 0.24967096296041524, 0.2481981003066959, 0.24703171731755016, 0.2459475046201557, 0.24484115655093708, 0.24079329977384878, 0.23982779521413672, 0.24433433166954388, 0.24878701829701053, 0.25319642120540126, 0.2571259064933801, 0.26110106390756366, 0.26515055999525333, 0.2687776885133644, 0.272400480811878, 0.2784885133835792, 0.277633439348945, 0.2663987715997962, 0.26024774196287703, 0.2564903555143584, 0.2597218316873521, 0.26701218184776787, 0.27988348374088934, 0.3002248990225164, 0.3237713229855621, 0.35496044847318026, 0.3877049092320624, 0.4224185100591648, 0.4555150479352217, 0.48740578558506076, 0.5138472355642341, 0.5404642007631928, 0.5672859337185896, 0.5878960373227444, 0.6074847292037137, 0.6301977540151567, 0.6509044488056265, 0.6694530743298556, 0.6845789022728556, 0.6987083306062132, 0.7115206570150624, 0.717683690678738, 0.717251054925835, 0.7155156718623061, 0.7151775290002281, 0.7222079211400038, 0.7231975519754633, 0.7186615251837193, 0.7181330411954732, 0.7197928595641754, 0.7275541646466995, 0.730476857781715, 0.7293192289012957, 0.7297637380974048, 0.7313892884212131, 0.7363937338768662, 0.7310371666234501, 0.7176303161140596, 0.7244854176495125, 0.7354819597897433, 0.752589606106537, 0.759204966913776, 0.7583154290579285, 0.7684395625515937, 0.7803391680909622, 0.7946402387100638, 0.8018290268642275, 0.804339937960915, 0.8163861166366534, 0.8290447780177105, 0.8418034397782219, 0.8589645228633377, 0.8787842110340802, 0.9004853923879637, 0.9223900801414388, 0.9445983077562975, 0.9496534493339193, 0.9452150590167165, 0.9473226349489308, 0.951837225207722, 0.9611515244650778, 0.9766341371651571, 0.9952043241725196, 1.0053627671546492, 1.0148019333780691, 1.0228702427201732, 1.0339107361243465, 1.0461016650805952, 1.049402035458862, 1.057685776298949, 1.0746627300557599, 1.0734433435891715, 1.0647654321611577, 1.0598543338746236, 1.0665629834137023, 1.0918632294737594, 1.105379201697975, 1.1155683889506878, 1.142359580961474, 1.1461511145689376, 1.1161331503235428, 1.103356711016767, 1.0990522065662278, 1.122390586946281, 1.139947456077115, 1.1496855195185423, 1.1489928835738448, 1.1474909319486062, 1.1663179176597558, 1.177717401149262, 1.1798662682354597, 1.1782183332866403, 1.1770184940788577, 1.1861683046265328, 1.1972496854574, 1.210548173048113, 1.2161231221781967, 1.2209082396052, 1.231983104959137, 1.2305488760307242, 1.2158758556534908, 1.2210569296489864, 1.228914485423375, 1.2291930003468206, 1.2275214231252487, 1.223946555970139, 1.2316630257172452, 1.2390882114426227, 1.2362700248226774, 1.242362747877623, 1.2564752896399751, 1.2461780867048293, 1.2352482787772077, 1.236353310953944, 1.23810690130679, 1.2403985547390761, 1.2414707068222828, 1.243817589151005, 1.2510440038227189, 1.2560726595295466, 1.2594215994560691, 1.265716815844647, 1.2716653725883822, 1.2756344651819795, 1.2783233609831735, 1.2801117045579877, 1.2604039876028266, 1.245323646307498, 1.24756726021471, 1.2515921246656625, 1.256768864468236, 1.2585863112601063, 1.2579710359264689, 1.2517879764584623, 1.2369976082443155, 1.217098385806663, 1.225480068785155, 1.2345695873725355, 1.2431217934125798, 1.2381539770318986, 1.2258346484599514, 1.17339987942472, 1.135883053656849, 1.1299496049104651, 1.1464990348548507, 1.174063730242261, 1.1822680396814746, 1.18818852083626, 1.1898334629934448, 1.1978721528901812, 1.208211557806303, 1.1981414791628966, 1.1912148375580298, 1.1896769451620963, 1.1894427274857808, 1.1900384486886393, 1.196383246153441, 1.20041941349298, 1.2008242980313188, 1.180265165493892, 1.1546613206110994, 1.166379179009079, 1.174594292221683, 1.1777460312059302, 1.1823455631422208, 1.1870578330375108, 1.1879607677638635, 1.1958975227171509, 1.2131905857081515, 1.2318523276863214, 1.2466263214681366, 1.2263767072369791, 1.2163210701752254, 1.2187553510867852, 1.2217181687185958, 1.2246591143204293, 1.2265061455970685, 1.2287596234474745, 1.231472105618257, 1.2308124699315255, 1.229442099224798, 1.228535503168583, 1.2278527610356134, 1.2274031229880065, 1.2253206238304046, 1.223037278794408, 1.2214064757000507, 1.222890950464094, 1.2273670809493005, 1.2252397503399508, 1.2222992236630414, 1.2208492113997251, 1.2206161673266875, 1.2214607439412246, 1.2227392977267388, 1.2220801846232683, 1.213912600869736, 1.2083645453592933, 1.2049544871648987, 1.2030014980555357, 1.2004837088920415, 1.1954871791648367, 1.191232884078422, 1.187536580991464, 1.1857200830005692, 1.1844864402778092, 1.1844435175403658, 1.178857478711149, 1.1694368184759745, 1.1637718495692828, 1.1598697552368729, 1.1599579653762464, 1.1591551256940509, 1.1577857907610571, 1.154711661265037, 1.1513484497096886, 1.1474941961267144, 1.1471556243704377, 1.1488678670130676, 1.1443806479341114, 1.140725427815854, 1.139306296282298, 1.1371459121180059, 1.1345895793435083, 1.1300664250802361, 1.125448582163741, 1.1206725143734222, 1.1206485768507728, 1.1228929012208815, 1.121025934615162, 1.117816383798644, 1.1121377125944536, 1.1061339559954437, 1.0999874335420698, 1.093892192820694, 1.090562969508197, 1.0918938842215324, 1.073854437225487, 1.049701785914876, 1.05616669708168, 1.0604645428906592, 1.0614112567029959, 1.0575029214125908, 1.0520143597118123, 1.048944584891088, 1.0463750884324707, 1.044516771998419, 1.0355717971525198, 1.023106889050859, 0.9989957914614015, 0.985801265290521, 0.9868901035925296, 0.9848488364706922, 0.9823270761275751, 0.9832251104675451, 0.971583131554889, 0.9448259580966486, 0.9409563161039123, 0.943353346124067, 0.9477687392386187, 0.9494603377103982, 0.9481254976902102, 0.9556196916928544, 0.964181414121452, 0.967501380665056, 0.9747491864475961, 0.986021417679871, 0.9952180314716156, 0.9982007191622735, 0.9736632481037966, 0.9769479024920653, 1.006521179349703, 1.0116370930359029, 1.0135532807295744, 1.0195037550602664, 1.022372766576816, 1.0225571703386223, 1.0178010539740963, 1.014337882018376, 1.0182027875038173, 1.0197239174394683, 1.019363183794996, 1.0179909913846992, 1.0199045021143893, 1.032912633130362, 1.028655617746253, 1.01163212086912, 1.005652024683847, 1.0010169221264908, 0.997313025830556, 0.9968224384566831, 0.9985177755195881, 1.0014294556178474, 1.0046427290599846, 1.0084281298717872, 1.0306657452559826, 1.0644359947479067, 1.1449574339167012, 1.1652200777207784, 1.0366581945978692, 0.9834663692985183, 0.9734803096305318, 0.9838591163254821, 0.9926480807442379, 0.9968938671355811, 0.9979491417713476, 0.9973303824612862, 1.0035492805109152, 1.0076602267117403, 1.0075511422927583, 1.006526231868194, 1.005085615077458, 1.0050430185006334, 1.0018968845271983, 0.9931432227576183, 0.9863899442064203, 0.980561679420573, 0.9761284553647798, 0.9716454582742032, 0.9670800413195816, 0.9640818536047938, 0.9616499084546234, 0.9583264785206023, 0.9546693316052574, 0.9505047986672911, 0.9410225206829896, 0.9309932571299476, 0.9371905026648528, 0.9392707520663047, 0.9355952010481758, 0.9366823116935022, 0.940017001827705, 0.9500366071153461, 0.9611162315092401, 0.9735601397528529, 0.973948964000294, 0.9735534733978173, 0.9932212325719573, 1.0105535275886302, 1.025116334135401, 1.031904859852798, 1.0364716337609945, 1.0394811014832905, 1.0439466615253947, 1.0500040513069535, 1.0640041576949455, 1.078696632694772, 1.0876448492735113, 1.0914077240585367, 1.0899430997994957, 1.04972757954173, 1.0157013439301268, 1.044777568312636, 1.0644969543470097, 1.0755174047296152, 1.0800264301978117, 1.0813849872867856, 1.074356955424126, 1.0686540097424435, 1.0640868522158806, 1.0611780613019943, 1.0584197262961772, 1.0553797136319234, 1.0511238524854762, 1.0459076472480187, 1.0470804565366192, 1.05449231280831, 1.0794919303118027, 1.0710003758868893, 1.0381516055140325, 1.0278331752133392, 1.024823079647246, 1.037034245682611, 1.0519526105440435, 1.0686818719175148, 1.0529674673286704, 1.0363702900446334, 1.0229071461750745, 1.0295321019146604, 1.0484987659108387, 1.0545715440692365, 1.0589593427980941, 1.0606798237546105, 1.056152894312911, 1.048106853950701, 1.057491961904594, 1.0619885794192103, 1.0553235360966668, 1.0528784372980875, 1.0526080338370127
]

# %%
reference_spectrum = np.array(reference_spectrum_TATP_FTIR)


IR_spectrum = np.array([float(i) for i in array_0])

correlation_threshold = 0.85

correlation, _ = pearsonr(reference_spectrum, IR_spectrum)

print(f"Pearson Correlation: {correlation:.4f}")

if correlation > correlation_threshold:
    print("TATP identified for spectrum 0 (IR sensor) by means of Pearson correlation")
else:
    print("TATP not detected for spectrum 0 (IR sensor) by means of Pearson correlation")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_FTIR)


IR_spectrum = np.array([float(i) for i in array_1])

correlation_threshold = 0.85

correlation, _ = pearsonr(reference_spectrum, IR_spectrum)

print(f"Pearson Correlation: {correlation:.4f}")

if correlation > correlation_threshold:
    print("TATP identified for spectrum 1 (IR sensor) by means of Pearson correlation")
else:
    print("TATP not detected for spectrum 1 (IR sensor) by means of Pearson correlation")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_FTIR)


IR_spectrum = np.array([float(i) for i in array_2])

correlation_threshold = 0.85

correlation, _ = pearsonr(reference_spectrum, IR_spectrum)

print(f"Pearson Correlation: {correlation:.4f}")

if correlation > correlation_threshold:
    print("TATP identified for spectrum 2 (IR sensor) by means of Pearson correlation")
else:
    print("TATP not detected for spectrum 2 (IR sensor) by means of Pearson correlation")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_FTIR)

IR_spectrum = np.array([float(i) for i in array_3])

correlation_threshold = 0.85

correlation, _ = pearsonr(reference_spectrum, IR_spectrum)

print(f"Pearson Correlation: {correlation:.4f}")

if correlation > correlation_threshold:
    print("TATP identified for spectrum 3 (IR sensor) by means of Pearson correlation")
else:
    print("TATP not detected for spectrum 3 (IR sensor) by means of Pearson correlation")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_FTIR)

IR_spectrum = np.array([float(i) for i in array_0])

correlation_threshold = 0.85

correlation, _ = kendalltau(reference_spectrum, IR_spectrum)

print(f"Correlazione di Kendall: {correlation:.4f}")

if correlation > correlation_threshold:
    print("TATP identified for spectrum 0 (IR sensor) by means of Kendall correlation")
else:
    print("TATP not detected for spectrum 0 (IR sensor) by means of Kendall correlation")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_FTIR)

IR_spectrum = np.array([float(i) for i in array_1])

correlation_threshold = 0.85

correlation, _ = kendalltau(reference_spectrum, IR_spectrum)

print(f"Correlazione di Kendall: {correlation:.4f}")

if correlation > correlation_threshold:
    print("TATP identified for spectrum 1 (IR sensor) by means of Kendall correlation")
else:
    print("TATP not detected for spectrum 1 (IR sensor) by means of Kendall correlation")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_FTIR)

IR_spectrum = np.array([float(i) for i in array_2])

correlation_threshold = 0.85

correlation, _ = kendalltau(reference_spectrum, IR_spectrum)

print(f"Correlazione di Kendall: {correlation:.4f}")

if correlation > correlation_threshold:
    print("TATP identified for spectrum 2 (IR sensor) by means of Kendall correlation")
else:
    print("TATP not detected for spectrum 2 (IR sensor) by means of Kendall correlation")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_FTIR)

IR_spectrum = np.array([float(i) for i in array_3])

correlation_threshold = 0.85

correlation, _ = kendalltau(reference_spectrum, IR_spectrum)

print(f"Correlazione di Kendall: {correlation:.4f}")

if correlation > correlation_threshold:
    print("TATP identified for spectrum 3 (IR sensor) by means of Kendall correlation")
else:
    print("TATP not detected for spectrum 3 (IR sensor) by means of Kendall correlation")

# %%
reference_spectrum.shape

# %%
reference_spectrum = np.array(reference_spectrum_TATP_FTIR)

IR_spectrum = np.array([float(i) for i in array_0])

correlation_threshold = 0.85

correlation, _ = spearmanr(reference_spectrum, IR_spectrum)

print(f"Spearman Correlation: {correlation:.4f}")

if correlation > correlation_threshold:
    print("TATP identified for spectrum 0 (IR sensor) by means of Spearman correlation")
else:
    print("TATP not detected for spectrum 0 (IR sensor) by means of Spearman correlation")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_FTIR)

IR_spectrum = np.array([float(i) for i in array_1])

correlation_threshold = 0.85

correlation, _ = spearmanr(reference_spectrum, IR_spectrum)

print(f"Spearman Correlation: {correlation:.4f}")

if correlation > correlation_threshold:
    print("TATP identified for spectrum 1 (IR sensor) by means of Spearman correlation")
else:
    print("TATP not detected for spectrum 1 (IR sensor) by means of Spearman correlation")


# %%
reference_spectrum = np.array(reference_spectrum_TATP_FTIR)

IR_spectrum = np.array([float(i) for i in array_2])

correlation_threshold = 0.85

correlation, _ = spearmanr(reference_spectrum, IR_spectrum)

print(f"Spearman Correlation: {correlation:.4f}")

if correlation > correlation_threshold:
    print("TATP identified for spectrum 2 (IR sensor) by means of Spearman correlation")
else:
    print("TATP not detected for spectrum 2 (IR sensor) by means of Spearman correlation")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_FTIR)

IR_spectrum = np.array([float(i) for i in array_3])

correlation_threshold = 0.85

correlation, _ = spearmanr(reference_spectrum, IR_spectrum)

print(f"Spearman Correlation: {correlation:.4f}")

if correlation > correlation_threshold:
    print("TATP identified for spectrum 3 (IR sensor) by means of Spearman correlation")
else:
    print("TATP not detected for spectrum 3 (IR sensor) by means of Spearman correlation")


# %%
reference_spectrum = np.array(reference_spectrum_TATP_FTIR)

IR_spectrum = np.array([float(i) for i in array_0])

if len(reference_spectrum) != len(IR_spectrum):
    raise ValueError("I dati di riferimento e di assorbanza devono avere la stessa lunghezza.")

reference_spectrum_2d = reference_spectrum.reshape(-1, 1)
absorbance_2d = IR_spectrum.reshape(-1, 1)

cca = CCA(n_components=1)
cca.fit(reference_spectrum_2d, absorbance_2d)

cca_score = cca.score(reference_spectrum_2d, absorbance_2d)
print(f"Canonical Correlation Analysis Score: {cca_score:.4f}")

correlation_threshold = 0.85

if cca_score > correlation_threshold:
    print("TATP identified for spectrum 0 (IR sensor) by means of Canonical Correlation Analysis")
else:
    print("TATP not detected for spectrum 0 (IR sensor) by means of Canonical Correlation Analysis")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_FTIR)

IR_spectrum = np.array([float(i) for i in array_1])

if len(reference_spectrum) != len(IR_spectrum):
    raise ValueError("I dati di riferimento e di assorbanza devono avere la stessa lunghezza.")

reference_spectrum_2d = reference_spectrum.reshape(-1, 1)
absorbance_2d = IR_spectrum.reshape(-1, 1)

cca = CCA(n_components=1)
cca.fit(reference_spectrum_2d, absorbance_2d)

cca_score = cca.score(reference_spectrum_2d, absorbance_2d)
print(f"Canonical Correlation Analysis Score: {cca_score:.4f}")

correlation_threshold = 0.85

if cca_score > correlation_threshold:
    print("TATP identified for spectrum 1 (IR sensor) by means of Canonical Correlation Analysis")
else:
    print("TATP not detected for spectrum 1 (IR sensor) by means of Canonical Correlation Analysis")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_FTIR)

IR_spectrum = np.array([float(i) for i in array_2])

if len(reference_spectrum) != len(IR_spectrum):
    raise ValueError("I dati di riferimento e di assorbanza devono avere la stessa lunghezza.")

reference_spectrum_2d = reference_spectrum.reshape(-1, 1)
absorbance_2d = IR_spectrum.reshape(-1, 1)

cca = CCA(n_components=1)
cca.fit(reference_spectrum_2d, absorbance_2d)

cca_score = cca.score(reference_spectrum_2d, absorbance_2d)
print(f"Canonical Correlation Analysis Score: {cca_score:.4f}")

correlation_threshold = 0.85

if cca_score > correlation_threshold:
    print("TATP identified for spectrum 2 (IR sensor) by means of Canonical Correlation Analysis")
else:
    print("TATP not detected for spectrum 2 (IR sensor) by means of Canonical Correlation Analysis")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_FTIR)

IR_spectrum = np.array([float(i) for i in array_3])

if len(reference_spectrum) != len(IR_spectrum):
    raise ValueError("I dati di riferimento e di assorbanza devono avere la stessa lunghezza.")

reference_spectrum_2d = reference_spectrum.reshape(-1, 1)
absorbance_2d = IR_spectrum.reshape(-1, 1)

cca = CCA(n_components=1)
cca.fit(reference_spectrum_2d, absorbance_2d)

cca_score = cca.score(reference_spectrum_2d, absorbance_2d)
print(f"Canonical Correlation Analysis Score: {cca_score:.4f}")

correlation_threshold = 0.85

if cca_score > correlation_threshold:
    print("TATP identified for spectrum 3 (IR sensor) by means of Canonical Correlation Analysis")
else:
    print("TATP not detected for spectrum 3 (IR sensor) by means of Canonical Correlation Analysis")

# %% [markdown]
# Reflectance IR for TATP

# %%
reference_spectrum_TATP_spray = [ 1.0382043692557827, 1.0486789801515655, 1.0457733811390817, 1.050401119707093, 1.0571011451870618, 1.0639653157521114, 1.0781815812080495, 1.1024261365884804, 1.1070567398918143, 1.1249732381424138, 1.114724128670026, 1.1408938921727736, 1.1484308887076322, 1.1621258633974003, 1.1799702787177815, 1.1762522873220627, 1.1998979114520858, 1.2067090130587974, 1.2126753656423037, 1.2224950331823052, 1.2342019958877029, 1.2428871504379775, 1.2585333011148636, 1.2655670573411257, 1.292975074072117, 1.2905026455482795, 1.2928228249357805, 1.305588295576953, 1.3124187956833677, 1.3421109791649661, 1.340202629835151, 1.3445277989565414, 1.3495463087984807, 1.3539962307773923, 1.3691721814595277, 1.3563157649951567, 1.3665519072381267, 1.370596900500348, 1.3719436684945387, 1.3645026452061884, 1.3725717183181159, 1.3821460176594695, 1.3748369261631985, 1.3698463656066389, 1.3694224158801391, 1.3575009893294063, 1.3569130811061074, 1.3422832301181244, 1.3280356720347117, 1.3168490062799143, 1.3057998483307245, 1.3101778148177043, 1.2995001262980357, 1.3035211811314045, 1.3077222819226468, 1.3055206116299647, 1.3085125904525254, 1.3024584932835759, 1.2981471941024867, 1.2926686332327504, 1.2861060179602233, 1.2878415713151858, 1.2900103530206484, 1.2944621248423425, 1.3023499886690204, 1.3035036977369099, 1.3111866935444227, 1.3222201806605411, 1.33325366777666, 1.3420638518079855, 1.3574072246674067, 1.3729997832555307, 1.3844882511404994, 1.3910003898759946, 1.3947819309549434, 1.4042380294474393, 1.4162381978440852, 1.4323542760576493, 1.4371101048636596, 1.4418659336696698, 1.4567648010384235, 1.4630775529940487, 1.4693903049496737, 1.4795800739603266, 1.4924268412189456, 1.5052736084775644, 1.5221149151510243, 1.5356266896939188, 1.5491384642368133, 1.5807225618267133, 1.6001671939685889, 1.6196118261104646, 1.630318118091938, 1.645751388317832, 1.6611846585437264, 1.6707068453291871, 1.6747392507641317, 1.6787716561990764, 1.6734369672616005, 1.6773543566193536, 1.6812717459771065, 1.6739761956511836, 1.6706174365691366, 1.667258677487091, 1.6634731349442493, 1.6526564415841587, 1.6414762443518798, 1.630799419414959, 1.6149126479168334, 1.5968813894134564, 1.5822834653823132, 1.5860131673216038, 1.5691560592171638, 1.55229895111273, 1.54476310728898, 1.535916973007782, 1.5270708387265806, 1.5270178837697443, 1.5179432435815947, 1.507197809543235, 1.500412193317719, 1.5012268000359603, 1.4910353093446989, 1.4808438186534412, 1.4781411289919415, 1.476360716007704, 1.4745803030234654, 1.472746975969547, 1.474561683942616, 1.4776610482238908, 1.4807604125051665, 1.4804028256164234, 1.4925515067691553, 1.5047001879218869, 1.5113262796473412, 1.514234894214658, 1.51842484607071, 1.5226147979267635, 1.5285463377304538, 1.5246014987727197, 1.520656659814987, 1.518859147988799, 1.5156928955597535, 1.5094811592005493, 1.5032694228413472, 1.4995327929985014, 1.488896591417839, 1.4782603898371727, 1.4675179284026667, 1.4542717615487846, 1.4396262732242837, 1.424980784899777, 1.4114198725423415, 1.4011429764204002, 1.391420462936874, 1.3816979494533446, 1.3775437881610706, 1.3748589661714434, 1.372174144181817, 1.3694893221921898, 1.369558619259195, 1.3703012614672585, 1.3710439036753217, 1.3707957248429308, 1.368633919043042, 1.3690521901087593, 1.3694704611744764, 1.3677700763028455, 1.3611125378425906, 1.3538641241328642, 1.3466157104231353, 1.343873695926587, 1.3367794362552368, 1.3282358156370762, 1.3196921950189124, 1.318410763870104, 1.310976999522994, 1.3032798339761278, 1.2955826684292584, 1.2875705438869482, 1.2901469958569614, 1.292723447826973, 1.2952998997969858, 1.2885320813911028, 1.2946374158037635, 1.300742750216422, 1.3068480846290829, 1.303129014320286, 1.3038946236235651, 1.304660232926844, 1.305425842230123, 1.2986636427791383, 1.2960817692685775, 1.293499895758018, 1.290918022247457, 1.293238511423602, 1.2933253442270674, 1.293412177030533, 1.2934990098339982, 1.2891362091800544, 1.295719888948159, 1.3023035687162632, 1.3088872484843652, 1.3135673721398249, 1.3109875968603595, 1.3081666525072433, 1.3053457081541282, 1.3040026218605838, 1.2963724352551913, 1.2873808044955042, 1.2783891737358204, 1.2726513672420146, 1.2578474887597437, 1.2372432532482631, 1.2166390177367903, 1.202177676961243, 1.1932630337699424, 1.1735264928773443, 1.1537899519847536, 1.1354206220476177, 1.1338827908268867, 1.1261069651479607, 1.1183311394690378, 1.1105553137901116, 1.104744033388026, 1.105327097194838, 1.1059101610016493, 1.1064932248084614, 1.1091946617021693, 1.1109920800138788, 1.1123607198867989, 1.1137293597597195, 1.1141697332739997, 1.1136522553434558, 1.1138779435671926, 1.1141036317909294, 1.114434233801983, 1.1193201203756786, 1.126627147794072, 1.1339341752124625, 1.1412412026308558, 1.14540258601768, 1.1576436436653856, 1.170419070108905, 1.1831944965524293, 1.1912011052880755, 1.1953800040227276, 1.2024823414556336, 1.209584678888542, 1.2166702516973047, 1.2198993378953897, 1.2255809052521847, 1.2312624726089816, 1.2369440399657765, 1.2372444822763562, 1.2383212861374437, 1.2401466127074823, 1.2419719392775204, 1.2424218101540117, 1.2365268847941497, 1.2289508155208109, 1.2213747462474749, 1.213798676974136, 1.211667658979443, 1.1989478146672792, 1.186227970355115, 1.1735081260429463, 1.1600255254285663, 1.146302440277449, 1.1335034048536936, 1.120704369429943, 1.1079053340061877, 1.0932181615566272, 1.0771470022021015, 1.0610758428475822, 1.0450046834930564, 1.0338523472681036, 1.02621519636652, 1.0163748020232615, 1.006534407679999, 0.9966940133367401, 0.9910013713111323, 0.9843111906549323, 0.9776210099987299, 0.9709308293425299, 0.9652260474641411, 0.9579052531549129, 0.9486481032394398, 0.93939095332397, 0.9301338034084969, 0.9227843711094498, 0.9116700190561933, 0.900555667002941, 0.8894413149496845, 0.8803898006274232, 0.8707889622122972, 0.8575474223886769, 0.8443058825650516, 0.8310643427414263, 0.8282408008556275, 0.8178830704101532, 0.8075004115199612, 0.7971177526297691, 0.7869928234783529, 0.7796302574025047, 0.7747846280622394, 0.7699389987219722, 0.765093369381707, 0.7589498575319955, 0.7528648903890087, 0.7468899230771406, 0.7409149557652749, 0.7355364181185664, 0.73289486568687, 0.7219414763319654, 0.7109880869770648, 0.7000346976221603, 0.6898759645232085, 0.6741069068079295, 0.655519101616734, 0.6369312964255315, 0.618343491234336, 0.6066680377309561, 0.5897701359579766, 0.5728722341849904, 0.5559743324120109, 0.5418497019989938, 0.5315770707412412, 0.5188532204928098, 0.5061293702443784, 0.4934055199959518, 0.4911328431182628, 0.48151033548826544, 0.4704223769235025, 0.45933441835874383, 0.449377695772188, 0.448274142775887, 0.4408857786057659, 0.43349741443564765, 0.42610905026552653, 0.4198939548531467, 0.41442224526640126, 0.4086828119633268, 0.4029433786602501, 0.39720394535717557, 0.3926998874839165, 0.3882163821839865, 0.3837328768840548, 0.37924937158412475, 0.3758632829343351, 0.3731410313504312, 0.36704657802300555, 0.36095212469558224, 0.3548576713681566, 0.3498309290423277, 0.3438998971355147, 0.33764663671973466, 0.33139337630395227, 0.3251904443248372, 0.3207608296420416, 0.31561846580683256, 0.31047610197162545, 0.30533373813641834, 0.3015226645080427, 0.3002178865504758, 0.2993311047691647, 0.2984443229878535, 0.297557541206542, 0.2942399914038234, 0.29600095613842786, 0.29776192087303305, 0.2995228856076375, 0.30059316129300245, 0.2998312952918286, 0.3000755868168869, 0.3003198783419452, 0.30056416986700346, 0.30102987107519774, 0.301906170265403, 0.30284460643525246, 0.3037830426051023, 0.3046807338613022, 0.30434429776677124, 0.3030292213154946, 0.3017141448642174, 0.3003990684129407, 0.29720230567646144, 0.29332834899376276, 0.2901836967601772, 0.28703904452659273, 0.28389439229300717, 0.2810765472014794, 0.27911678949889646, 0.27715703179631423, 0.27519727409373135, 0.2721070341536218, 0.2708333282611172, 0.2725834312533842, 0.2743335342456519, 0.27608363723791957, 0.27579423388415486, 0.2788376959913763, 0.2818811580985966, 0.284924620205818, 0.28715663424526056, 0.28902820576395044, 0.29234626589307267, 0.2956643260221962, 0.29898238615131845, 0.29994255812010673, 0.30324122531428405, 0.3065398925084626, 0.3098385597026399, 0.31322017643879047, 0.31768727792488494, 0.3229662442828278, 0.3282452106407687, 0.33352417699871156, 0.33855288416025425, 0.3453534738470589, 0.3521540635338609, 0.35895465322066555, 0.365749242667386, 0.37107693311212286, 0.3755246654905986, 0.37997239786907594, 0.38442013024755334, 0.39163863295357737, 0.39521945494448324, 0.39880027693539044, 0.40238109892629764, 0.40518742553653364, 0.4084173549100142, 0.41187584368752117, 0.4153343324650282, 0.4193175894069775, 0.4230771192900002, 0.4240453397395863, 0.42501356018917247, 0.42598178063875825, 0.42538256628915433, 0.4269789942056525, 0.42857542212215133, 0.4301718500386495, 0.42916054336855763, 0.43242993559574183, 0.437450974878445, 0.44247201416114623, 0.447757505898251, 0.45741637416623737, 0.4706247813640702, 0.4838331885618983, 0.4970415957597312, 0.5154059940973463, 0.5354212520727499, 0.5554365100481459, 0.5754517680235495, 0.6030746331942544, 0.6160237156967745, 0.6289727981992944, 0.6419218807018194, 0.6550451999569453, 0.6826081750933239, 0.7127007450874807, 0.7427933150816488, 0.7706348746861081, 0.7867475831172175, 0.7991526726094099, 0.811557762101607, 0.8243236715780431, 0.8429696258890628, 0.8651197679809712, 0.8872699100728714, 0.9091207793751296, 0.9286322898338271, 0.9468902683160941, 0.9651482467983546, 0.9816959353390012, 0.9968627100102978, 1.0177821495934403, 1.038701589176575, 1.0580155035304506, 1.076194018303387, 1.099485585205428, 1.12277715210746, 1.1466533276315496, 1.1577243086081188, 1.157350890739523, 1.156977472870927, 1.1618931945232742, 1.1661296722152337, 1.1648556588363994, 1.1635816454575652, 1.16934325317145, 1.1600264461770873, 1.1454035506843379, 1.1307806551915884, 1.1017415154894394, 1.0675127985330195, 1.0332840815765867, 0.9990553646201539, 0.9808700435829688, 0.9344507538847424, 0.8880314641864983, 0.8356100316222977, 0.7758326058259167, 0.7296560659049861, 0.6834795259840729, 0.6261858135828782, 0.5964696023031134, 0.5707309051687857, 0.5449922080344675, 0.5530069777674824, 0.5669243686429694, 0.5808417595184615, 0.6051736562443383, 0.6514224953647005, 0.6993320623060345, 0.7472416292473864, 0.7946527170880455, 0.84513275028538, 0.8956127834826955, 0.9557698861651882, 0.9901554210365854, 1.0189389875200772, 1.0481161013178901, 1.0749503099789501, 1.0958113778555925, 1.116672445732227, 1.13854954777279, 1.1487370050422114, 1.1589244623116326, 1.1742917904732897, 1.193193340518297, 1.2120666727250409, 1.2303837488233904, 1.2429199302298333, 1.2542628123230948, 1.2718738201808268, 1.288390839849311, 1.2954944166810158, 1.2999620129174132, 1.308067678860014, 1.3241066839134972, 1.3314782397358118, 1.3293176514939373, 1.3381086776401234, 1.3384444399169217, 1.3395580940261291, 1.3463325079742439, 1.3603386442838112, 1.3716262157345858, 1.3824526171220062, 1.394583838250727, 1.3908239746493658, 1.387064111048006, 0.5723758666359158, 0.5682267138669732, 0.5677018710686612, 0.566815934194774, 0.5597909507196137, 0.5541745231929588, 0.5537477488333136, 0.5484862344948443, 0.5555832920824341, 0.5549219723440466, 0.5582718598866241, 0.5625567167999869, 0.5667574843501274, 0.5799441710327277, 0.5826811118461325, 0.5892220399081572, 0.5927816705001985, 0.5998372131811698, 0.589879350658199, 0.6028998728696798, 0.5969254932408615, 0.6045054035662857, 0.6068695103164783, 0.6382198430223402, 0.655506184005899, 0.6957001949077946, 0.7165688003383232, 0.7467353510671882, 0.7807400443382938, 0.8042709677797026, 0.8107153459415122, 0.824575605898992, 0.8233591067087518, 0.8487656536373127, 0.8441069537848163, 0.8587914110196271, 0.8523201161623227, 0.8588303949495357, 0.8638683944107522, 0.8606056039165076, 0.8600033013894066, 0.8656846257534706, 0.8633981075209648, 0.8750728922695589, 0.8726764666060125, 0.8702507914309232, 0.8746343737901189, 0.8708875619512157, 0.8653095883108154, 0.8733121517281972, 0.8539376495343267, 0.8662941319961724, 0.8718331212974079, 0.8699840929456587, 0.8684074620822732, 0.8699534474405526, 0.8725085824564053, 0.8823662142945805, 0.8888341445503611, 0.8948798347512982, 0.8988491725901743, 0.9128319465189548, 0.9203711728354605, 0.9271072384321268, 0.9345245011284955, 0.9374619688708069, 0.9425259886417616, 0.9539288501787108, 0.9587530641446899, 0.9556493850443573, 0.9633432821992249, 0.9596979471072681, 0.9564120717005963, 0.9720328009107182, 0.9748615894074132, 0.9828629946545978, 1.0067346129155117, 1.0324108578776046, 1.0725022373766107, 1.0571211444735653, 1.0417400515705144, 1.0526354676101655, 1.0386051878118006, 1.0292833439486406, 1.046428577473111, 1.0416374520259954, 1.040774417084575, 1.0415835801788655, 1.0386069068978545, 1.0270968278923953, 1.0277460957024176, 1.0325258535614519, 1.0249751027081586, 1.0299552381039885, 1.0353555588567214, 1.021695124038397, 1.022229750151314, 1.0227643762642313, 1.0186018855542258, 1.0136060516572833, 1.008610217760343, 1.028518737994416, 1.0327113482428536, 1.0369039584912898, 1.0407503128258175, 1.0373088578662932, 1.0338674029067687, 1.0547451287182201, 1.050040288016077, 1.0453354473139322, 1.0681619204737083, 1.0659352376320215, 1.0637085547903355, 1.0818710937105025, 1.082131015742018, 1.0823909377735337, 1.0767487929845272, 1.0838717523304224, 1.0909947116763175, 1.0718324615865154, 1.077843901718207, 1.083855341849901, 1.0688508659823603, 1.0712192264688756, 1.075969657128, 1.0688508169501125, 1.0653914303073133, 1.0664045342881447, 1.0656374420569574, 1.0654344972789647, 1.067805538821436, 1.0730017230590994, 1.0839776522207687, 1.0806973858775841, 1.077417119534398, 1.1108599262015892, 1.1094508132812908, 1.1080417003609928, 1.1185466068684242, 1.1165847665959778, 1.1146229263235312, 1.1244079172872812, 1.131525934868062, 1.1342985452033418, 1.1401976366154642, 1.1516550280045055, 1.158049690668934, 1.1644443533333602, 1.1757029879462069, 1.1794651255013888, 1.183227263056571, 1.191698965630184, 1.1927278284098761, 1.1930034225546118, 1.199033066666921, 1.2060203588843392, 1.2061502641419302, 1.2062801693995209, 1.219409962075571, 1.2122607251348962, 1.2051114881942189, 1.2110061560528298, 1.1901880376759277, 1.1663605737113676, 1.1496347842843653, 1.1428412972756852, 1.1290117213178925, 1.1151821453600996, 1.1103402976893215, 1.1083797499306545, 1.1064192021719863, 1.102131913936313, 1.0975271648144709, 1.09357238985902, 1.091806483291583, 1.1018165121662074, 1.0877285019955638, 1.07364049182492, 1.0930039380926193, 1.0813198036588443, 1.0649339182901516, 1.0608564226074806, 1.0799256545874143, 1.073875603328336, 1.0678255520692579, 1.0867011107909978, 1.0896778434216232, 1.09265457605225, 1.0943621607969198, 1.1009494584436768, 1.1142272288594108, 1.1275049992751398, 1.1086339299133137, 1.128181803849046, 1.1477296777847856, 1.155017973948666, 1.150158647986111, 1.1551183698298486, 1.1600780916735842, 1.1550862531266741, 1.1496606302762329, 1.144235007425792, 1.1423452306430146, 1.1444730580306373, 1.143830993340144, 1.143188928649651, 1.1415209682876115, 1.136635095773197, 1.1317492232587845, 1.12613111536246, 1.1206180173588747, 1.1170289992385092, 1.1134399811181424, 1.1060674555871646, 1.101374164246707, 1.0967708320725271, 1.0916993173990763, 1.08622218663197, 1.08259378689823, 1.0789653871644884, 1.0738475298417545, 1.0708265777110477, 1.0681632778580648, 1.0653042022845935, 1.0618568694599795, 1.060514327195275, 1.05917178493057, 1.0561601807322782, 1.0546498227753935, 1.0539478891081775, 1.0532459554409621, 1.0489161553035893, 1.0486621568808414, 1.0484081584580935, 1.0475169852489215, 1.0449371272915322, 1.0419204248694736, 1.038903722447416, 1.0421626399064072, 1.0357124530049062, 1.029262266103405, 1.0268084578540557, 1.0254354524157552, 1.0163826435068488, 1.0073298345979458, 1.009241803450981, 1.0005439252119799, 0.9911772585468147, 0.9824756004512719, 0.977739744459867, 0.973776993267712, 0.9698142420755553, 0.9697590721825906, 0.9666135423283131, 0.9622306381603868, 0.9583761051841942, 0.967252371340563, 0.9627025089162072, 0.9581526464918496, 0.9638637879011285, 0.9678415391372135, 0.9659797920157717, 0.9641180448943307, 0.9952520199705404, 0.9903601507262326, 0.9854682814819249, 0.9872821276114593, 0.987079549703124, 0.9767332271140672, 0.9663869045250145, 1.0068483367643568, 0.9837694997721559, 0.9606906627799547, 0.9714159183837475, 1.0085726027378963, 0.9736145033744457, 0.9386564040110085, 0.9966038010669293, 0.9742597700574417, 0.9467594971894552, 0.932768244900408, 0.9513533989433304, 0.9374578809080014, 0.9235623628726778, 0.9697029259123158, 0.9689661831909844, 0.9606250811752166, 0.956325189663026, 0.9638816192074776, 0.9567672253830719, 0.9496528315586636, 1.012071084358014, 1.002813383070408, 0.9814288030386279, 0.9776797626546898, 1.0178767377034406, 0.9920409112546674, 0.9662050848058845, 0.991339008266979, 0.9814541204245923, 0.967413478194543, 0.9591389573349233, 0.9657328420770633, 0.9657416487106165, 0.9657504553441697, 0.9609421867554875, 0.9670954921995629, 0.9732487976436407, 0.960380409278918, 0.9465253487503893, 0.9598805448972276, 0.973235741044066, 0.9431307632869615, 0.9621986118938489, 0.9812664605007435, 0.9816338281635921, 0.9766394840182543, 0.975898662714521, 0.9780928658824066, 0.992681484218061, 0.987161996721604, 0.981642509225145, 1.032587499598737, 1.0137353558390552, 0.9948832120793664, 0.993801870114862, 0.9955507352435005, 0.9840722162164531, 0.9740610187171214, 0.9848850559168977, 0.9826151783337407, 0.9803453007505828, 0.9795519169265043, 0.9852121221359753, 0.9908723273454483, 0.9958410987892558, 1.0109950000491017, 1.0299988698028533, 1.0316299053924478, 1.0040529929772026, 1.0290873919171244, 1.0541217908570555, 1.005149215282771, 1.026439882872353, 1.0477305504619432, 1.0055131580994028, 1.0237889909120979, 1.042064823724786, 1.0281506733283161, 1.027757459633254, 1.0354157157869526, 1.0378705201129426, 1.0353722011338722, 1.0348936622286957, 1.0425446550409563, 1.0572494534934338, 1.0539496044222452, 1.0490388971596378, 1.0380464102039362, 1.0359738629243977, 1.033787762533472, 1.029922976327857, 1.0302137612761308, 1.030504546224405, 1.0342715077518148, 1.0317968171024354, 1.029322126453057, 1.0414479101930108, 1.0467528412759026, 1.0520577723587927, 1.0494673504097232, 1.060166980371972, 1.070514717041609, 1.057466492592008, 1.0852658950132024, 1.1030566436230969, 1.065484967867558, 1.091577785153862, 1.0948009184363925, 1.0687226963969059, 1.0903967350925188, 1.0737721273050058, 1.0722664641510296, 1.1035383391533526, 1.0684377356349442, 1.076621843912856, 1.093638726475619, 1.082777191352373, 1.0483595639730554, 1.0145765134820093, 1.070763832259051, 1.0352451926983035, 1.1126991219827798, 1.1376096177971131, 1.041350035097063, 1.2162335955251866, 1.1104892813507439, 1.0077904791278072, 1.0483279609425364, 1.031765316532709, 0.9959626856559901, 1.0163477118685622, 1.0475962435836903, 1.0011367066518855, 1.0227384316984764, 0.9970663205771623, 1.0079745156030369, 1.0354776573105458, 1.0036038224636885, 1.024858511027403, 1.006031677048161
]

# %%
reference_spectrum = np.array(reference_spectrum_TATP_spray)


# Converti i dati di assorbanza da stringhe a float e in un array NumPy
IR_spectrum = np.array([float(i) for i in array_0])

# Imposta la soglia di correlazione
correlation_threshold = 0.85

# Calcola la correlazione di Pearson tra i dati di assorbanza e il riferimento
correlation, _ = pearsonr(reference_spectrum, IR_spectrum)

# Stampa il valore della correlazione
print(f"Correlazione di Pearson: {correlation:.4f}")

# Verifica se la correlazione è sopra la soglia e stampa il messaggio
if correlation > correlation_threshold:
    print("TATP identified for spectrum 0 (IR sensor) by means of Pearson correlation")
else:
    print("TATP not detected for spectrum 0 (IR sensor) by means of Pearson correlation")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_spray)


# Converti i dati di assorbanza da stringhe a float e in un array NumPy
IR_spectrum = np.array([float(i) for i in array_1])

# Imposta la soglia di correlazione
correlation_threshold = 0.85

# Calcola la correlazione di Pearson tra i dati di assorbanza e il riferimento
correlation, _ = pearsonr(reference_spectrum, IR_spectrum)

# Stampa il valore della correlazione
print(f"Correlazione di Pearson: {correlation:.4f}")

# Verifica se la correlazione è sopra la soglia e stampa il messaggio
if correlation > correlation_threshold:
    print("TATP identified for spectrum 1 (IR sensor) by means of Pearson correlation")
else:
    print("TATP not detected for spectrum 1 (IR sensor) by means of Pearson correlation")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_spray)


# Converti i dati di assorbanza da stringhe a float e in un array NumPy
IR_spectrum = np.array([float(i) for i in array_2])

# Imposta la soglia di correlazione
correlation_threshold = 0.85

# Calcola la correlazione di Pearson tra i dati di assorbanza e il riferimento
correlation, _ = pearsonr(reference_spectrum, IR_spectrum)

# Stampa il valore della correlazione
print(f"Correlazione di Pearson: {correlation:.4f}")

# Verifica se la correlazione è sopra la soglia e stampa il messaggio
if correlation > correlation_threshold:
    print("TATP identified for spectrum 2 (IR sensor) by means of Pearson correlation")
else:
    print("TATP not detected for spectrum 2 (IR sensor) by means of Pearson correlation")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_spray)


# Converti i dati di assorbanza da stringhe a float e in un array NumPy
IR_spectrum = np.array([float(i) for i in array_3])

# Imposta la soglia di correlazione
correlation_threshold = 0.85

# Calcola la correlazione di Pearson tra i dati di assorbanza e il riferimento
correlation, _ = pearsonr(reference_spectrum, IR_spectrum)

# Stampa il valore della correlazione
print(f"Correlazione di Pearson: {correlation:.4f}")

# Verifica se la correlazione è sopra la soglia e stampa il messaggio
if correlation > correlation_threshold:
    print("TATP identified for spectrum 3 (IR sensor) by means of Pearson correlation")
else:
    print("TATP not detected for spectrum 3 (IR sensor) by means of Pearson correlation")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_spray)

IR_spectrum = np.array([float(i) for i in array_0])

correlation_threshold = 0.85

correlation, _ = kendalltau(reference_spectrum, IR_spectrum)

print(f"Correlazione di Kendall: {correlation:.4f}")

if correlation > correlation_threshold:
    print("TATP identified for spectrum 0 (IR sensor) by means of Kendall correlation")
else:
    print("TATP not detected for spectrum 0 (IR sensor) by means of Kendall correlation")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_spray)

IR_spectrum = np.array([float(i) for i in array_1])

correlation_threshold = 0.85

correlation, _ = kendalltau(reference_spectrum, IR_spectrum)

print(f"Correlazione di Kendall: {correlation:.4f}")

if correlation > correlation_threshold:
    print("TATP identified for spectrum 1 (IR sensor) by means of Kendall correlation")
else:
    print("TATP not detected for spectrum 1 (IR sensor) by means of Kendall correlation")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_spray)

IR_spectrum = np.array([float(i) for i in array_2])

correlation_threshold = 0.85

correlation, _ = kendalltau(reference_spectrum, IR_spectrum)

print(f"Correlazione di Kendall: {correlation:.4f}")

if correlation > correlation_threshold:
    print("TATP identified for spectrum 2 (IR sensor) by means of Kendall correlation")
else:
    print("TATP not detected for spectrum 2 (IR sensor) by means of Kendall correlation")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_spray)

IR_spectrum = np.array([float(i) for i in array_3])

correlation_threshold = 0.85

correlation, _ = kendalltau(reference_spectrum, IR_spectrum)

print(f"Correlazione di Kendall: {correlation:.4f}")

if correlation > correlation_threshold:
    print("TATP identified for spectrum 3 (IR sensor) by means of Kendall correlation")
else:
    print("TATP not detected for spectrum 3 (IR sensor) by means of Kendall correlation")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_spray)

IR_spectrum = np.array([float(i) for i in array_0])

correlation_threshold = 0.85

correlation, _ = spearmanr(reference_spectrum, IR_spectrum)

print(f"Spearman Correlation: {correlation:.4f}")

if correlation > correlation_threshold:
    print("TATP identified for spectrum 0 (IR sensor) by means of Spearman correlation")
else:
    print("TATP not detected for spectrum 0 (IR sensor) by means of Spearman correlation")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_spray)

IR_spectrum = np.array([float(i) for i in array_1])

correlation_threshold = 0.85

correlation, _ = spearmanr(reference_spectrum, IR_spectrum)

print(f"Spearman Correlation: {correlation:.4f}")

if correlation > correlation_threshold:
    print("TATP identified for spectrum 1 (IR sensor) by means of Spearman correlation")
else:
    print("TATP not detected for spectrum 1 (IR sensor) by means of Spearman correlation")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_spray)

IR_spectrum = np.array([float(i) for i in array_2])

correlation_threshold = 0.85

correlation, _ = spearmanr(reference_spectrum, IR_spectrum)

print(f"Spearman Correlation: {correlation:.4f}")

if correlation > correlation_threshold:
    print("TATP identified for spectrum 2 (IR sensor) by means of Spearman correlation")
else:
    print("TATP not detected for spectrum 2 (IR sensor) by means of Spearman correlation")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_spray)

IR_spectrum = np.array([float(i) for i in array_3])

correlation_threshold = 0.85

correlation, _ = spearmanr(reference_spectrum, IR_spectrum)

print(f"Spearman Correlation: {correlation:.4f}")

if correlation > correlation_threshold:
    print("TATP identified for spectrum 3 (IR sensor) by means of Spearman correlation")
else:
    print("TATP not detected for spectrum 3 (IR sensor) by means of Spearman correlation")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_spray)

IR_spectrum = np.array([float(i) for i in array_0])

if len(reference_spectrum) != len(IR_spectrum):
    raise ValueError("I dati di riferimento e di assorbanza devono avere la stessa lunghezza.")

reference_spectrum_2d = reference_spectrum.reshape(-1, 1)
absorbance_2d = IR_spectrum.reshape(-1, 1)

cca = CCA(n_components=1)
cca.fit(reference_spectrum_2d, absorbance_2d)

cca_score = cca.score(reference_spectrum_2d, absorbance_2d)
print(f"Canonical Correlation Analysis Score: {cca_score:.4f}")

correlation_threshold = 0.85

if cca_score > correlation_threshold:
    print("TATP identified for spectrum 0 (IR sensor) by means of Canonical Correlation Analysis")
else:
    print("TATP not detected for spectrum 0 (IR sensor) by means of Canonical Correlation Analysis")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_spray)

IR_spectrum = np.array([float(i) for i in array_1])

if len(reference_spectrum) != len(IR_spectrum):
    raise ValueError("I dati di riferimento e di assorbanza devono avere la stessa lunghezza.")

reference_spectrum_2d = reference_spectrum.reshape(-1, 1)
absorbance_2d = IR_spectrum.reshape(-1, 1)

cca = CCA(n_components=1)
cca.fit(reference_spectrum_2d, absorbance_2d)

cca_score = cca.score(reference_spectrum_2d, absorbance_2d)
print(f"Canonical Correlation Analysis Score: {cca_score:.4f}")

correlation_threshold = 0.85

if cca_score > correlation_threshold:
    print("TATP identified for spectrum 1 (IR sensor) by means of Canonical Correlation Analysis")
else:
    print("TATP not detected for spectrum 1 (IR sensor) by means of Canonical Correlation Analysis")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_spray)

IR_spectrum = np.array([float(i) for i in array_2])

if len(reference_spectrum) != len(IR_spectrum):
    raise ValueError("I dati di riferimento e di assorbanza devono avere la stessa lunghezza.")

reference_spectrum_2d = reference_spectrum.reshape(-1, 1)
absorbance_2d = IR_spectrum.reshape(-1, 1)

cca = CCA(n_components=1)
cca.fit(reference_spectrum_2d, absorbance_2d)

cca_score = cca.score(reference_spectrum_2d, absorbance_2d)
print(f"Canonical Correlation Analysis Score: {cca_score:.4f}")

correlation_threshold = 0.85

if cca_score > correlation_threshold:
    print("TATP identified for spectrum 2 (IR sensor) by means of Canonical Correlation Analysis")
else:
    print("TATP not detected for spectrum 2 (IR sensor) by means of Canonical Correlation Analysis")

# %%
reference_spectrum = np.array(reference_spectrum_TATP_spray)

IR_spectrum = np.array([float(i) for i in array_3])

if len(reference_spectrum) != len(IR_spectrum):
    raise ValueError("I dati di riferimento e di assorbanza devono avere la stessa lunghezza.")

reference_spectrum_2d = reference_spectrum.reshape(-1, 1)
absorbance_2d = IR_spectrum.reshape(-1, 1)

cca = CCA(n_components=1)
cca.fit(reference_spectrum_2d, absorbance_2d)

cca_score = cca.score(reference_spectrum_2d, absorbance_2d)
print(f"Canonical Correlation Analysis Score: {cca_score:.4f}")

correlation_threshold = 0.85

if cca_score > correlation_threshold:
    print("TATP identified for spectrum 3 (IR sensor) by means of Canonical Correlation Analysis")
else:
    print("TATP not detected for spectrum 3 (IR sensor) by means of Canonical Correlation Analysis")

# %% [markdown]
# Import training data for GC-QEPAS and train the model

# %%
QEPAS = pd.read_excel("/Users/giorgiofelizzato/Desktop/RISEN project/data fusion/Moduli python data fusion-final version/TATP data for DF/TATP_GCQEPAS_dataset.xlsx", sheet_name='QEPAS', index_col=0, header=0)
QEPAS.head()

# %%
GC = pd.read_excel("/Users/giorgiofelizzato/Desktop/RISEN project/data fusion/Moduli python data fusion-final version/TATP data for DF/TATP_GCQEPAS_dataset.xlsx", sheet_name='GC', index_col=0, header=0)
GC.head()

# %%
# select only numerical attributes
X_QEPAS = QEPAS.iloc[:, 1:]
X_QEPAS

# %%
# select only numerical attributes
X_GC = GC.iloc[:, 1:]
X_GC

# %%
#Selection of classes from the spectra database
GCQEPAS_class = QEPAS.loc[:, 'Class'].values
GCQEPAS_class

# %%
GCQEPAS_class_dataframe = pd.DataFrame(GCQEPAS_class, columns=['Class'])
GCQEPAS_class_dataframe

# %%
# It is necessary to convert the column names as string to select them
QEPAS.columns = QEPAS.columns.astype(str) # to make the colnames as text

# %%
# Write the SNV function (is, actually, like autoscaling by row)
def snv_QEPAS(input_data):
  
    # Define a new array and populate it with the corrected data  
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
 
        # Apply correction
        output_data[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
 
    return output_data

# %%
# Compute the SNV on spectra
Xsnv_QEPAS = snv_QEPAS(X_QEPAS.values)
Xsnv_QEPAS

# %%
# Preprocessing with Savitzki-Golay - smoothing, defining the window, the order and the use of derivatives
X_savgol_QEPAS = savgol_filter(X_QEPAS, 7, polyorder = 2, deriv=0)
X_savgol_QEPAS

# %%
# We can also combine the preprocessing strategies together: Savitzki-Golay - smoothing + SNV
X_savgol_QEPAS = savgol_filter(X_QEPAS, 7, polyorder = 2, deriv=0)
X_snv_savgol_QEPAS = snv_QEPAS(X_savgol_QEPAS)
X_snv_savgol_QEPAS

# %%
processed_dataframe_QEPAS = pd.DataFrame(Xsnv_QEPAS, columns=QEPAS.columns[1:])
processed_dataframe_QEPAS

# %%
# X training set
X_train_QEPAS = pd.concat([GCQEPAS_class_dataframe, processed_dataframe_QEPAS], axis = 1)
X_train_QEPAS

# %%
X_train_QEPAS_cut = X_train_QEPAS.drop(X_train_QEPAS.columns[0], axis=1)

# Mostra le prime righe dopo aver eliminato la colonna
X_train_QEPAS_cut

# %%
# Define the PCA object
pca = PCA()
 
# Run PCA producing the reduced variable Xreg and select the first 10 components
pca = PCA(n_components=5)
Xreg = pca.fit_transform(X_train_QEPAS_cut)

# %%
print ("Proportion of Variance Explained : ", pca.explained_variance_ratio_)  
   
out_sum = np.cumsum(pca.explained_variance_ratio_)  
print ("Cumulative Prop. Variance Explained: ", out_sum)

# %%
# Run PCA producing the pca_model with a proper number of components
pca = PCA(n_components=5)
pca_model = pca.fit_transform(X_train_QEPAS_cut)

# %%
# Define the class vector (discrete/categorical variable)
Classes_QEPAS = GCQEPAS_class_dataframe.astype('category')
Classes_QEPAS

# %%
# Prepare the Scores dataframe (and concatenate the original 'Region' variable)
scores = pd.DataFrame(data = pca_model, columns = ['PC1','PC2','PC3', 'PC4', 'PC5'])
scores.index = processed_dataframe_QEPAS.index
scores = pd.concat([scores, X_train_QEPAS.Class], axis = 1)
print(scores)

# %%
# Prepare the loadings dataframe
loadings = pd.DataFrame(pca.components_.T, columns=['PC1','PC2','PC3', 'PC4', 'PC5'], index=X_train_QEPAS_cut.columns)
loadings["Attributes"] = loadings.index
loadings

# %%
# View the scores plot using plotly library
fig = px.scatter(scores, x="PC1", y="PC2", color="Class", hover_data=['Class'], hover_name=processed_dataframe_QEPAS.index)
fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='Black')
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='Black')
fig.update_layout(
    height=600,
    width=800,
    title_text='Scores Plot colored by Substance')
fig.show()

# %%
scores

# %%
X_pca = pd.DataFrame(scores)
X_pca =X_pca.drop(columns='Class')
X_pca

# %%
# Supponiamo che X_pca sia un DataFrame o una matrice numpy
X_pca = np.array(X_pca)

# Calcola la matrice di covarianza dei punteggi PCA
cov_matrix = np.cov(X_pca, rowvar=False)

# Calcola l'inversa della matrice di covarianza
inv_cov_matrix = inv(cov_matrix)

# Assicura che il centroide sia un vettore numpy 1-D
centroid = np.mean(X_pca, axis=0)

# Calcola la distanza Euclidea tra ogni punto e il centroide
euclidean_distances_train = cdist(X_pca, [centroid], metric='euclidean').flatten()

# Calcola la distanza di Mahalanobis tra ogni punto (vettore 1-D) e il centroide (vettore 1-D)
mahalanobis_distances_train = [mahalanobis(x, centroid, inv_cov_matrix) for x in X_pca]


# %%
centroid

# %%
euclidean_distances_train

# %%
mahalanobis_distances_train

# %%
euclidean_threshold = 5.5  
mahalanobis_threshold = 6.5  

# %% [markdown]
# import new data of GC-QEPAS (json file)

# %%
# GC retention time
file_path_GC = '/Users/giorgiofelizzato/Desktop/RISEN project/data fusion/Moduli python data fusion-final version/TATP data for DF/GC-QEPAS data/test json/measurement_2d-e02c6571-d98e-4d82-8d2c-497f7546df52-2024-08-06T09_11_03.731796.json'

# Open the file and load its content
with open(file_path_GC, 'r') as file:
    data_GC = json.load(file)

# Print the data to verify
print(data_GC)

# %%
def find_x_value(json_data):
    if "data" in json_data:
        data = json_data["data"]
        if "columns" in data:
            columns = data["columns"]
            for column in columns:
                if column[0] == "x":
                    return column[1:]  
    return None

x_values = find_x_value(data_GC)

if x_values is not None:
    RT = pd.DataFrame(x_values, columns=["RT"])
    print(RT)
else:
    print("tr not found")

# %%
RT_unknown=RT
print(RT_unknown)

# %%
RT_unknown_str = RT.iloc[0]['RT']
try:
    RT_unknown = float(RT_unknown_str)
except ValueError:
    print("Il valore RT non è valido.")
    RT_unknown = None

target_value = 107.22
tolerance = 2.28
lower_bound = target_value - tolerance
upper_bound = target_value + tolerance

if RT_unknown is not None and lower_bound <= RT_unknown <= upper_bound:
    print("TATP detected by means of retention time (GC sensor)")
else:
    print("TATP NOT detected by means of retention time (GC sensor)")

# %%
#intensity of the GC peak
data = data_GC

def find_ia_value(json_data):
    if "data" in json_data:
        data = json_data["data"]
        if "columns" in data:
            columns = data["columns"]
            for column in columns:
                if column[0] == "IA":
                    return column[1]  
    return None

IA = find_ia_value(data)

if IA is not None:
    print(f"IA: {IA}")
else:
    print("'IA' not found")

# %%
# QEPAS spectra
file_path_QEPAS = '/Users/giorgiofelizzato/Downloads/measurement_2d-c4b87a83-6abc-451b-8609-dcea8c671014-2024-08-06T14_02_46.148778.json'

# Open the file and load its content
with open(file_path_QEPAS, 'r') as file:
    data_QEPAS = json.load(file)

# Print the data to verify
print(data_QEPAS)

# %%
data = data_QEPAS

def extract_x_values(json_data):
    if "data" in json_data and "columns" in json_data["data"]:
        for column in json_data["data"]["columns"]:
            if column[0] == "x":
                return column[1:]
    return []

wavelenght = extract_x_values(data)

if wavelenght:
    print("wavelenght:", wavelenght)
else:
    print("'x' not found")

# %%
data = data_QEPAS

def extract_y_values(json_data):
    if "data" in json_data and "columns" in json_data["data"]:
        for column in json_data["data"]["columns"]:
            if column[0] == "y":
                return column[1:]
    return []

absorbance = extract_y_values(data)

if absorbance:
    print("absorbance:", absorbance)
else:
    print(" 'y' NOt found")

# %%
absorbance

# %%
# Assuming y_values is your input data list
y_values = np.array(absorbance)

# Apply Savitzky-Golay filter
X_savgol_unknown = savgol_filter(y_values, 7, polyorder=2, deriv=0)

# Reshape to 2D if necessary
if X_savgol_unknown.ndim == 1:
    X_savgol_unknown = X_savgol_unknown.reshape(1, -1)

# Apply SNV
X_snv_savgol_spectra_unknown = snv_QEPAS(X_savgol_unknown)
X_snv_savgol_spectra_unknown

# %%
# Create a new DataFrame with the processed numerical attributes
processed_dataframe_spectra_unknown = pd.DataFrame(X_snv_savgol_spectra_unknown, columns=QEPAS.columns[1:])
processed_dataframe_spectra_unknown.head()

# %%
pca_model_newdataQEPAS = pca.transform(processed_dataframe_spectra_unknown)

# %%
pca_model_newdataQEPAS

# %%
new_sample_pca = pca_model_newdataQEPAS

euclidean_distances = np.linalg.norm(new_sample_pca - centroid, axis=1)

mahalanobis_distances = [mahalanobis(x, centroid, inv_cov_matrix) for x in new_sample_pca]

def predict_simca(euclidean_distances, mahalanobis_distances, euclidean_threshold, mahalanobis_threshold):
    is_class = [(euc <= euclidean_threshold) and (mah <= mahalanobis_threshold)
                for euc, mah in zip(euclidean_distances, mahalanobis_distances)]
    return is_class

is_in_class = predict_simca(euclidean_distances, mahalanobis_distances, euclidean_threshold, mahalanobis_threshold)

print(f"TATP identified: {is_in_class} by means of SIMCA (QEPAS sensor)")

# %%
euclidean_distances

# %%
mahalanobis_distances

# %%
reference_spectrum_QEPAS = [
    -5.4, -26.9, -48.3, -52.3, -56.0, -58.9, -60.4, -62.5, -64.6, -52.2, -46.2, -49.1, -44.9,
    -40.2, -37.2, -42.5, -52.9, -61.5, -57.3, -53.0, -46.1, -40.8, -41.5, -45.6, -57.9, -63.2,
    -70.1, -70.7, -67.7, -72.9, -76.7, -84.0, -88.0, -90.2, -90.2, -85.5, -87.4, -92.9, -92.5,
    -88.9, -88.8, -92.6, -95.8, -94.6, -94.3, -97.0, -100.7, -114.6, -113.3, -118.3, -111.8,
    -108.5, -107.0, -102.5, -97.9, -101.0, -93.2, -95.1, -99.6, -105.1, -101.2, -109.4, -117.8,
    -117.0, -109.0, -117.2, -130.1, -127.5, -140.1, -163.0, -178.5, -194.6, -230.3, -250.4,
    -267.2, -285.9, -345.5, -385.0, -423.9, -452.2, -491.6, -523.7, -564.0, -603.4, -568.9,
    -524.6, -494.1, -423.5, -324.7, -188.1, -26.6, 151.6, 381.8, 684.8, 1068.8, 1553.4, 2174.7,
    2848.8, 3554.0, 4219.0, 4778.8, 5182.5, 5454.4, 5601.7, 5618.5, 5555.7, 5535.5, 5480.5,
    5264.4, 4881.3, 4336.7, 3713.4, 3064.0, 2471.7, 1821.2, 1354.3, 1074.3, 926.1, 875.6, 867.6,
    884.9, 905.2, 890.5, 830.2, 704.6, 562.3, 429.0, 309.4
]


# %%
reference_spectrum = np.array(reference_spectrum_QEPAS)

absorbance = np.array([float(i) for i in absorbance])

correlation_threshold = 0.85

correlation, _ = pearsonr(reference_spectrum, absorbance)

print(f"Correlazione di Pearson: {correlation:.4f}")

if correlation > correlation_threshold:
    print("TATP identified by means of Pearson Correlation (QEPAS sensor)")
else:
     print("TATP not identified by means of Pearson Correlation (QEPAS sensor)")

# %%
reference_spectrum = np.array(reference_spectrum_QEPAS)

absorbance = np.array([float(i) for i in absorbance])

correlation_threshold = 0.85

correlation, _ = kendalltau(reference_spectrum, absorbance)

print(f"Correlazione di Kendall: {correlation:.4f}")

if correlation > correlation_threshold:
    print("TATP identified by means of Kendall Correlation (QEPAS sensor)")
else:
     print("TATP not identified by means of Kendall Correlation (QEPAS sensor)")


# %%
reference_spectrum = np.array(reference_spectrum_QEPAS)

absorbance = np.array([float(i) for i in absorbance])

correlation_threshold = 0.85

correlation, _ = spearmanr(reference_spectrum, absorbance)

print(f"Correlazione di Spearman: {correlation:.4f}")

if correlation > correlation_threshold:
    print("TATP identified by means of Spearman Correlation (QEPAS sensor)")
else:
     print("TATP not identified by means of Spearman Correlation (QEPAS sensor)")

# %%
reference_spectrum = np.array(reference_spectrum_QEPAS)

absorbance = np.array([float(i) for i in absorbance])

if len(reference_spectrum) != len(absorbance):
    raise ValueError("I dati di riferimento e di assorbanza devono avere la stessa lunghezza.")

reference_spectrum_2d = reference_spectrum.reshape(-1, 1)
absorbance_2d = absorbance.reshape(-1, 1)

cca = CCA(n_components=1)
cca.fit(reference_spectrum_2d, absorbance_2d)

cca_score = cca.score(reference_spectrum_2d, absorbance_2d)
print(f"Canonical Correlation Analysis Score: {cca_score:.4f}")

correlation_threshold = 0.85

if cca_score > correlation_threshold:
    print("TATP identified by means of Canonical Correlation Analysis Correlation (QEPAS sensor)")
else:
     print("TATP not identified by means of Canonical Correlation Analysis Correlation (QEPAS sensor)")


