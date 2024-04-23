from curve_fit import poly_data
import numpy as np
import csv
import json

iterations = 5
num_nodes = 10
time_start = 0
time_stop = 10
num_points = 200

metadata = {
    'num_nodes' : num_nodes,
    'adj_matrix_length' : [[0 for i in range(num_nodes)] for i in range(num_nodes)],
    'iterations' : iterations,
    'time_start' : time_start,
    'time_stop' : time_stop,
    'num_points' : num_points    
}

data = {
    'time' : [],
    'source' : [],
    'dest' : [],    # Each value is added in as [source, dest]
    'edge_rate' : [],
}

def generate_data():
    for i in range(num_nodes**2):
        source = i // num_nodes
        dest = i % num_nodes

        if np.random.randint(0, 10) >= 4 or source == dest:
            continue

        metadata['adj_matrix_length'][source][dest] = np.random.randint(50, 100)

        coeffs = [np.random.uniform(30, 100)] + [np.random.uniform(0, 0.08) * np.random.choice([-1, 0, 1]) for k in range(3)] + np.random.uniform(0, 0.08) * np.random.choice([0, 1])
            
        for j in range(iterations):
            timestamps, edge_weights = poly_data(coeffs, time_start, time_stop, num_points, 0, 0.5)
            data['time'] += list(timestamps)
            data['edge_rate'] += list(edge_weights)
            data['source'] += [i // num_nodes] * num_points
            data['dest'] += [i % num_nodes] * num_points
            #data['path'] += list([i // num_nodes, i % num_nodes] * num_points)

    print(metadata)

    # Specify the data and metadata file paths
    data_filepath = 'data.csv'
    metadata_filepath = 'metadata.json'

    # Write the dictionary to a CSV file
    with open(data_filepath, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=data.keys())
        
        # Write header
        writer.writeheader()
        
        # Write rows
        for i in range(len(data['time'])):
            writer.writerow({key: data[key][i] for key in data.keys()})

    with open(metadata_filepath, 'w') as json_file:
        json.dump(metadata, json_file)

