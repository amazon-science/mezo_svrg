import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.size': 12})
import numpy as np
import pickle
import torch
import argparse
import utils
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser for Model Fine-tuning')

    parser.add_argument('--folder_path', type=str, default='results_demo', help='Path to results folder')
    parser.add_argument('--plot_title', type=str, default='Demo', help='Title for plot')
    parser.add_argument('--x_lim_low_iteration', type=float, default=0, help='Lower limit on x-axis for iteration plot')
    parser.add_argument('--x_lim_high_iteration', type=float, default=10000, help='Upper limit on x-axis for iteration plot')
    parser.add_argument('--x_lim_low_time', type=float, default=1e4, help='Lower limit on x-axis for time plot')
    parser.add_argument('--x_lim_high_time', type=float, default=1e8, help='Upper limit on x-axis for time plot')
    parser.add_argument('--x_lim_low_query', type=float, default=1e5, help='Lower limit on x-axis for query plot')
    parser.add_argument('--x_lim_high_query', type=float, default=1e9, help='Upper limit on x-axis for query plot')
    parser.add_argument('--y_lim_high', type=float, default=1, help='Upper limit on y-axis')
    parser.add_argument('--y_lim_low', type=float, default=0, help='Lower limit on y-axis')



    args = parser.parse_args()
    return args

def open_pickle_files_in_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Iterate through the files and open pickle files
    results_iteration = {}
    results_time = {}
    results_query = {}
    for filename in file_list:
        if filename.endswith('.pickle'):
            file_path = os.path.join(folder_path, filename)
            try:
                print(file_path)
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)

                key = f"{data['Algorithm']}_BS{data['BS']}_LR{data['LR']}"
                results_iteration[key] = data['Tr_Loss']
                results_time[key] = data['Time']
                results_query[key] = data['Query']
            except Exception as e:
                print(f"Error opening {filename}: {e}")
    return results_iteration, results_time, results_query
                
def open_pickle_files_in_folder_mu(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Iterate through the files and open pickle files
    results_iteration = {}
    results_time = {}
    results_query = {}
    for filename in file_list:
        if filename.endswith('.pickle'):
            file_path = os.path.join(folder_path, filename)
            try:
                print(file_path)
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
                parts = filename.split('_')
                for part in parts:
                    if "perturbationscale" in part:
                        value_str = part.split("perturbationscale")[1]
                        value_str = value_str.split('.pickle')[0]
                        key = f'$\mu = {value_str}$'
                #key = f"{data['Algorithm']}_BS{data['BS']}_LR{data['LR']}"
                results_iteration[key] = data['Tr_Loss']
                results_time[key] = data['Time']
                results_query[key] = data['Query']
            except Exception as e:
                print(f"Error opening {filename}: {e}")
    return results_iteration, results_time, results_query

def open_pickle_files_in_folder_q(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Iterate through the files and open pickle files
    results_iteration = {}
    results_time = {}
    results_query = {}
    for filename in file_list:
        if filename.endswith('.pickle'):
            file_path = os.path.join(folder_path, filename)
            try:
                print(file_path)
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
                parts = filename.split('_')
                for part in parts:
                    if "q" in part:
                        value_str = part.split("q")[1]
                        key = f'$q = {value_str}$'
                #key = f"{data['Algorithm']}_BS{data['BS']}_LR{data['LR']}"
                results_iteration[key] = data['Tr_Loss']
                results_time[key] = data['Time']
                results_query[key] = data['Query']
            except Exception as e:
                print(f"Error opening {filename}: {e}")
    return results_iteration, results_time, results_query


if __name__ == "__main__":
    args = parse_arguments()
    
    path = args.folder_path
    title = args.plot_title
    x_lim_low_iteration = args.x_lim_low_iteration
    x_lim_high_iteration = args.x_lim_high_iteration
    x_lim_low_time = args.x_lim_low_time
    x_lim_high_time = args.x_lim_high_time
    x_lim_low_query = args.x_lim_low_query
    x_lim_high_query = args.x_lim_high_query
    y_lim_high = args.y_lim_high
    y_lim_low = args.y_lim_low
    
    iteration, time, query = open_pickle_files_in_folder(path)
    
    utils.plot_results(title, iteration, loss='Training Loss', lim_x=x_lim_high_iteration, lim_y=y_lim_high)
    utils.plot_results_time(title + ' Time', iteration, time, loss='Training Loss', lim_x=x_lim_high_time, lim_y=y_lim_high)
    utils.plot_results_query(title + ' Query', iteration, query, loss='Training Loss', lim_x=x_lim_high_query, lim_y=y_lim_high)
    
    
    
    
