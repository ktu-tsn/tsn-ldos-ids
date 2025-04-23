# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:17:40 2024

@author: Mustafa Topsakal
"""

import classification, data_handling
import numpy as np
import os, time

fe_start_index = data_handling.fe_start_index

#Definition input-output folder and result file
normal_scenario_path = "../tsn_dataset/raw_dataset/normal_scenario/"
attack_scenarios_path = "../tsn_dataset/raw_dataset/attack_scenarios/"
ml_ds_path = "../tsn_dataset/ml_dataset/"
results_path = "../output/results.txt"

start_time = time.perf_counter()

#Take full path non_attack scenario
ns_file = os.listdir(os.path.abspath(normal_scenario_path))[0];

if ns_file:
    ns_full_path = os.path.join(os.path.abspath(normal_scenario_path), ns_file)
else:
    print("No file found in the directory.")

#Read non_attack scenario, apply data proccessing and feature extraction
print("Non-attack scenario is being created...")
dataNormal, labelsNormal, normal_count_zeros, normal_count_ones = data_handling.data_processing(ns_full_path)        
dataNormal = data_handling.feature_extraction(dataNormal)
labelsNormal = labelsNormal[fe_start_index:]
print(f"Number of Zeros:{normal_count_zeros} - Number Of Ones:{normal_count_ones} - Total Number:{normal_count_zeros + normal_count_ones}")

#After applying feature extraction add the data to the ML_dataset folder  
data_handling.write_output(dataNormal, labelsNormal, os.path.join(ml_ds_path, ns_file))

#Write information about normal scenario to file
with open(os.path.abspath(results_path), "a+", encoding='utf-8') as result_file:
    result_file.write(ns_file + "\n")
    result_file.write(f"Number of Zeros:{normal_count_zeros} - Number Of Ones:{normal_count_ones} - Total Number:{normal_count_zeros + normal_count_ones}\n")
    result_file.write("---------------------------------------\n")

#Run all scenarios with all ML algorithms
for attack_sce in os.listdir(os.path.abspath(attack_scenarios_path)):
    
    print(f"\nAttack dataset is being created for: {attack_sce}")
    attack_sce_path = os.path.join(os.path.abspath(attack_scenarios_path), attack_sce)

    dataAttack, labelsAttack, attack_count_zeros, attack_count_ones = data_handling.data_processing(attack_sce_path)
    dataAttack = data_handling.feature_extraction(dataAttack)
    labelsAttack = labelsAttack[fe_start_index:]
    
    print(f"Number of Zeros:{attack_count_zeros} - Number Of Ones:{attack_count_ones} - Total Number:{attack_count_zeros + attack_count_ones}")

    #After applying feature extraction add the data to the ML_dataset folder  
    data_handling.write_output(dataAttack, labelsAttack, os.path.join(ml_ds_path, attack_sce))

    #Write information about attack scenario to file
    with open(os.path.abspath(results_path), "a+", encoding='utf-8') as result_file:
        result_file.write("\n" + attack_sce + "\n")
        result_file.write(f"Number of Zeros:{attack_count_zeros} - Number Of Ones:{attack_count_ones} - Total Number:{attack_count_zeros + attack_count_ones}")
        result_file.write("\n---------------------------------------\n")

    #Include normal scenario
    #all_data = dataNormal + dataAttack
    #all_labels = labelsNormal + labelsAttack
    
    #Considering the data imbalance: If only attack scenario will be used
    all_data = dataAttack
    all_labels = labelsAttack
    
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
        
    for classifier_index in range(len(classification.ml_methods)):
        model_name = classification.ml_methods[classifier_index][0]
        print(f"\nTraining and testing process is starting with {model_name} in {attack_sce}")
        with open(os.path.abspath(results_path), "a+", encoding='utf-8') as output_file:
            output_file.write(attack_sce + "\n")
            output_file.write(classification.classify(all_data, all_labels, classifier_index))
            output_file.write("---------------------------------------\n")
            print(f"Results have been written to the file {results_path}")
     
end_time = time.perf_counter()
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)

print(f"Total time: {int(hours)} hour, {int(minutes)} minute, {seconds:.2f} second")