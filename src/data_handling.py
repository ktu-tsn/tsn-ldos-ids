# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:18:06 2024

@author: Mustafa Topsakal
"""

fe_start_index = 3 #Depth selected for feature extraction

def data_processing(file_path):
    data = []
    labels = []
    count_zeros= 0
    count_ones = 0
    with open(file_path, 'r') as file:
        for line in file:
          parts = line.strip().split(',')
          timestamp = float(parts[0])
          stream_no = int(parts[1])
          source = string_to_decimal(parts[2])
          destination = string_to_decimal(parts[3])
          source_mac = mac_to_decimal(parts[4])
          dest_mac = mac_to_decimal(parts[5])
          packet_length = int(parts[6])
          label = int(parts[7])
                      
          #Add all the data.
          data.append([timestamp, stream_no,source,destination,source_mac,dest_mac,packet_length])
          labels.append(label)
          if(label == 1):
              count_ones += 1
          else:
              count_zeros += 1           
                   
    return data, labels, count_zeros, count_ones


def feature_extraction(data):
    fe_data = []

    for i in range(fe_start_index, len(data)):
        timestamp = data[i][0]
        last_remote_timestamp = "{:.12f}".format(float(timestamp) - float(data[i-1][0]))
        stream_id = data[i][1]
        prev_stream_id = data[i-1][1]
        prev_prev_stream_id = data[i-2][1]
        prev_prev_prev_stream_id = data[i-3][1]
        source_id = data[i][2]
        destination_id = data[i][3]
        packet_source_mac = data[i][4]
        packet_dest_mac = data[i][5]
        packet_length = data[i][6]
                  
        fe_data.append([timestamp, last_remote_timestamp, stream_id, prev_stream_id, prev_prev_stream_id, prev_prev_prev_stream_id, 
                        source_id, destination_id, packet_source_mac, packet_dest_mac, packet_length])
    
    return fe_data


def write_output(data, labels, output_file):
    with open(output_file, 'w+') as file:
        for i in range(len(data)):
            #Merge data and labels and write.
            file.write(','.join(map(str, data[i] + [labels[i]])) + '\n')
            
def mac_to_decimal(mac_address):
    return int(''.join(mac_address.split('-')), 16)

def string_to_decimal(string):
    if "Cam1" in string:
        return 1
    elif "Cam2" in string:
        return 2
    elif "Cam3" in string:
        return 3
    elif "DA_Cam" in string:
        return 4
    elif "HU" in string:
        return 5
    elif "RSE" in string:
        return 6
    elif "Telematics" in string:
        return 7
    elif "CU" in string:
        return 8
    elif "CD_DVD" in string:
        return 9
    elif "Cam4" in string:
        return 10
    elif "SW1" in string:
        return 11
    elif "SW2" in string:
        return 12
    else:
        return None