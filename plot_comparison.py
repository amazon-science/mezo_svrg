import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.size': 12})
import numpy as np
import pickle
import torch

def convert_to_cpu_float(dictionary):
    """
    Convert PyTorch tensors in a dictionary from cuda:0 to floats on CPU.

    Parameters:
    - dictionary: Dictionary with keys and PyTorch tensors/lists of tensors on cuda:0.

    Returns:
    - Converted dictionary with tensors/lists as floats on CPU.
    """
    converted_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            converted_dict[key] = value.cpu().float()
        elif isinstance(value, list):
            converted_dict[key] = [item.cpu().numpy() for item in value]
        else:
            converted_dict[key] = value  # Non-tensor values remain unchanged

    return converted_dict


def average_across_batch(arr, epochs):
    arr_np = np.array(arr)
    arr_avg = np.mean(np.array(np.split(arr_np, epochs)), axis=1)
    #print(arr_avg.shape)
    return np.reshape(arr_avg, (1, arr_avg.shape[0]))

def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_results(title, d, loss="Train Loss: ", lim_x=1000, lim_y=3):
    plt.figure()
    for k, val in d.items():

        v = np.array([val])
        #print(v.shape)
        means, mins, maxes = np.mean(v, axis=0), np.amin(v, axis=0), np.amax(v, axis=0)
        #print(means.shape)
        means = moving_average(means)
        #print('hi', means)
        plt.plot(range(len(means)), means, linewidth=2, linestyle='solid', markersize=12, label=k)
        #plt.fill_between(range(len(val['Tr_Loss'])), mins, maxes, alpha=0.5)
        
    plt.title(title)
    plt.yscale('log')
    plt.xlim(0, lim_x)
    plt.ylim(0, lim_y)
    plt.ylabel(loss)
    plt.xlabel('Step')
    plt.legend(bbox_to_anchor=(0.65, 1.0), loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.savefig(title + '.pdf')

def plot_results_time(title, d_y, d_x, loss="Train Loss: ", lim_x=1000, lim_y=3):
    plt.figure()
    for k, val in d_y.items():
        v = np.array([val])
        v_x = np.cumsum(np.array([d_x[k][:-1]]))*(1e-17)
        print(v_x[-1])
        
        means, mins, maxes = np.mean(v, axis=0), np.amin(v, axis=0), np.amax(v, axis=0)
        means = moving_average(means)
        l = len(means)
        plt.plot(v_x[:l], means, linewidth=2, linestyle='solid', markersize=12, label=k)
        #plt.fill_between(range(len(val['Tr_Loss'])), mins, maxes, alpha=0.5)
        
    plt.title(title)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(15, lim_x)
    plt.ylim(0.9, lim_y)
    plt.ylabel(loss)
    plt.xlabel('Time (s)')
    plt.legend(bbox_to_anchor=(0.0, 1), loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.savefig(title + '.pdf')
    
def plot_results_query(title, d_y, d_x, loss="Train Loss: ", lim_x=1000, lim_y=3):
    plt.figure()
    for k, val in d_y.items():
        v = np.array([val])
        v_x = np.cumsum(np.array([d_x[k][:-1]]))
        
        means, mins, maxes = np.mean(v, axis=0), np.amin(v, axis=0), np.amax(v, axis=0)
        means = moving_average(means)
        l = len(means)
        plt.plot(v_x[:l], means, linewidth=2, linestyle='solid', markersize=12, label=k)
        #plt.fill_between(range(len(val['Tr_Loss'])), mins, maxes, alpha=0.5)
        
    plt.title(title)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(1e4, lim_x)
    plt.ylim(1.085, lim_y)
    plt.ylabel(loss)
    plt.xlabel('Queries')
    plt.legend(bbox_to_anchor=(0.0, 1), loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.savefig(title + '.pdf')
    
def plot_norm(title, d, loss="Train Loss: ", lim_x=1000, lim_y=3):
    plt.figure()
    for k, val in d.items():

        v = np.array([val])
        #print(v.shape)
        means, mins, maxes = np.mean(v, axis=0), np.amin(v, axis=0), np.amax(v, axis=0)
        #print(means.shape)
        means = moving_average(means)
        #print('hi', means)
        plt.plot(range(len(means)), means, linewidth=2, linestyle='solid', markersize=12, label=k)
        #plt.fill_between(range(len(val['Tr_Loss'])), mins, maxes, alpha=0.5)
        
    plt.title(title)
    plt.yscale('log')
    plt.xlim(0, lim_x)
    plt.ylim(0, lim_y)
    plt.ylabel(loss)
    plt.xlabel('Step')
    plt.legend(bbox_to_anchor=(0.0, 1), loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.savefig(title + '.pdf')


    
if __name__ == "__main__":    
    # result_dict = {}
    # path1 = 'results/LS_FO.pickle' 
    # path2 = 'results/LS_ZO.pickle'
    # path3 = 'results/LS_ZO_lr1e5.pickle'
    # path4 = 'results/LS_ZO_n10000_d=1000_lr5e4_bs10000.pickle'
    
    # with open(path1, 'rb') as f:
    #     data1 = pickle.load(f)
    # result_dict['FO, lr=1e-3'] = data1
    
    # with open(path2, 'rb') as f:
    #     data2 = pickle.load(f)
    # result_dict['ZO, lr=1e-4'] = data2
    
    # with open(path3, 'rb') as f:
    #     data3 = pickle.load(f)
    # result_dict['ZO, lr=1e-5'] = data3
    
    # with open(path4, 'rb') as f:
    #     data4 = pickle.load(f)
    # result_dict['ZO, lr=5e-4, BS=10000'] = data4
        
    # #plot_results('LS comparison', result_dict, lim_x=500, lim_y=2)

    # result_dict = {}
    # path1 = 'results/LS_FO_n1000_d=1000.pickle' 
    # path2 = 'results/LS_ZO_n1000_d=1000_lr1e4.pickle'
    # path3 = 'results/LS_ZO_n1000_d=1000_lr1e5.pickle'
    # path4 = 'results/LS_ZO_n1000_d=1000_lr5e4_bs1000.pickle'
    
    # with open(path1, 'rb') as f:
    #     data1 = pickle.load(f)
    # result_dict['FO, lr=1e-3'] = data1
    
    # with open(path2, 'rb') as f:
    #     data2 = pickle.load(f)
    # result_dict['ZO, lr=1e-4'] = data2
    
    # with open(path3, 'rb') as f:
    #     data3 = pickle.load(f)
    # result_dict['ZO, lr=1e-5'] = data3
    
    # with open(path4, 'rb') as f:
    #     data4 = pickle.load(f)
    # result_dict['ZO, lr=5e-4, bs=1000'] = data4
        
    # #plot_results('LS comparison (n=1000, d=1000)', result_dict, lim_x=400, lim_y=2)
    
    # result_dict = {}
    # path1 = 'results/LS_FO_n100_d=1000.pickle' 
    # path2 = 'results/LS_ZO_n100_d=1000_lr1e4.pickle'
    # path3 = 'results/LS_ZO_n100_d=1000_lr1e5.pickle'
    # path4 = 'results/LS_ZO_n100_d=1000_lr5e4.pickle'
    # path5 = 'results/LS_ZO_n100_d=1000_lr2e4.pickle'
    
    # with open(path1, 'rb') as f:
    #     data1 = pickle.load(f)
    # result_dict['FO, lr=1e-3'] = data1
    
    # with open(path2, 'rb') as f:
    #     data2 = pickle.load(f)
    # result_dict['ZO, lr=1e-4'] = data2
    
    # with open(path3, 'rb') as f:
    #     data3 = pickle.load(f)
    # result_dict['ZO, lr=1e-5'] = data3
    
    # with open(path4, 'rb') as f:
    #     data4 = pickle.load(f)
    # result_dict['ZO, lr=2e-4'] = data4
    
    # with open(path5, 'rb') as f:
    #     data5 = pickle.load(f)
    # result_dict['ZO, lr=5e-4'] = data5
        
    # #plot_results('LS comparison (n=100, d=1000)', result_dict, lim_x=500, lim_y=2)
    
    
    # result_dict = {}
    # path1 = 'results/MNIST_FO_lr1e3_bs64.pickle'
    # path2 = 'results/MNIST_ZO_lr1e3_bs64.pickle'
    # path3 = 'results/MNIST_ZO_lr1e3_bs128.pickle'
    # path4 = 'results/MNIST_ZO_lr1e3_bs256.pickle'
    # path5 = 'results/MNIST_ZO_lr1e3_bs512.pickle'
    # path6 = 'results/MNIST_ZO_lr1e3_bs1024.pickle'
    # path7 = 'results/MNIST_ZO_lr1e2_bs2048.pickle'
    
    # with open(path1, 'rb') as f:
    #     data1 = pickle.load(f)
    # result_dict['FO, lr=1e-3'] = data1
    
    # with open(path2, 'rb') as f:
    #     data2 = pickle.load(f)
    # result_dict['ZO, lr=1e-3, BS=64'] = data2
    
    # with open(path3, 'rb') as f:
    #     data3 = pickle.load(f)
    # result_dict['ZO, lr=1e-3, BS=128'] = data3
    
    # with open(path4, 'rb') as f:
    #     data4 = pickle.load(f)
    # result_dict['ZO, lr=1e-3, BS=256'] = data4
    
    # with open(path5, 'rb') as f:
    #     data5 = pickle.load(f)
    # result_dict['ZO, lr=1e-3, BS=512'] = data5
    
    # with open(path6, 'rb') as f:
    #     data6 = pickle.load(f)
    # result_dict['ZO, lr=1e-3, BS=1024'] = data6
    
    # with open(path7, 'rb') as f:
    #     data6 = pickle.load(f)
    # result_dict['ZO, lr=1e-2, BS=2048'] = data6
        
    # #plot_results('MNIST comparison', result_dict, lim_x=1000, lim_y=3)
    
    # # long term comparison
    # result_dict = {}
    # path1 = 'results/LS_SGD_n1000_d=1000.pickle'
    # path2 = 'results/LS_ZO_grad_cosine_bs1000.pickle'
    # path3 = 'results/LS_ZO_grad_cosine_bs512.pickle'
    # path4 = 'results/LS_ZO_grad_cosine_bs256.pickle'
    # path5 = 'results/LS_ZO_grad_cosine_bs128.pickle'
    # path6 = 'results/LS_ZO_SVRG_Coord_Rand_5000.pickle'
    
    # with open(path1, 'rb') as f:
    #     data1 = pickle.load(f)
    # result_dict['GD, lr=1e-3'] = data1
    
    # with open(path2, 'rb') as f:
    #     data2 = pickle.load(f)
    # result_dict['ZO, lr=1e-4, BS=1000'] = data2['Tr_Loss']
    # #print('1000: ', len(result_dict['ZO, lr=1e-4, BS=1000']))
    
    # with open(path3, 'rb') as f:
    #     data3 = pickle.load(f)
    # #result_dict['ZO, lr=1e-4, BS=512'] = average_across_batch(data3['Tr_Loss'], 15000).tolist()[0]
    
    # with open(path4, 'rb') as f:
    #     data4 = pickle.load(f)
    # #result_dict['ZO, lr=1e-4, BS=256'] = average_across_batch(data4['Tr_Loss'], 15000).tolist()[0]

    # with open(path5, 'rb') as f:
    #     data5 = pickle.load(f)
    # result_dict['ZO, lr=1e-4, BS=128'] = average_across_batch(data5['Tr_Loss'], 15000).tolist()[0]
    
    # with open(path6, 'rb') as f:
    #     data6 = pickle.load(f)
    # result_dict['ZO-Rand, lr=1e-4, BS=1000'] = data6['Tr_Loss']
            
    # #plot_results('LS with linear model (n=1000, d=1000)', result_dict, lim_x=5000, lim_y=1000)
    
    # # long term comparison
    # result_dict = {}
    # result_dict['ZO, lr=1e-4, BS=1000'] = data2['Grad_Dot']
    # result_dict['ZO, lr=1e-4, BS=512'] = average_across_batch(data3['Grad_Dot'], 15000).tolist()[0]
    # result_dict['ZO, lr=1e-4, BS=256'] = average_across_batch(data4['Grad_Dot'], 15000).tolist()[0]
    # result_dict['ZO, lr=1e-4, BS=128'] = average_across_batch(data5['Grad_Dot'], 15000).tolist()[0]
    # #plot_results('LS with linear model (n=1000, d=1000): Dot Product', result_dict, loss='Grad Dot Product: ', lim_x=15000, lim_y=1000)

    # result_dict = {}
    # result_dict['ZO, lr=1e-4, BS=1000'] = data2['Abs_Proj']
    # result_dict['ZO, lr=1e-4, BS=512'] = average_across_batch(data3['Abs_Proj'], 15000).tolist()[0]
    # result_dict['ZO, lr=1e-4, BS=256'] = average_across_batch(data4['Abs_Proj'], 15000).tolist()[0]
    # result_dict['ZO, lr=1e-4, BS=128'] = average_across_batch(data5['Abs_Proj'], 15000).tolist()[0]
    # #plot_results('LS with linear model (n=1000, d=1000): Grad Norm', result_dict, loss='Grad Norm', lim_x=15000, lim_y=1000)

    # result_dict = {}
    # path1 = 'results_LS_n100_d100/LS_ZO_n100_d100_bs32_lr1e3.pickle'
    # path2 = 'results_LS_n100_d100/LS_ZO_n100_d100_bs16_lr1e3.pickle'
    # path3 = 'results_LS_n100_d100/LS_ZO_n100_d100_bs8_lr1e4.pickle'
    # path4 = 'results_LS_n100_d100/LS_ZO_SVRG_Coord_Rand_FD_n100_d100_bs32_lr4e2.pickle'
    # path5 = 'results_LS_n100_d100/LS_ZO_SVRG_Coord_Rand_FD_n100_d100_bs16_lr1e2.pickle'
    # path6 = 'results_LS_n100_d100/LS_ZO_SVRG_Coord_Rand_FD_n100_d100_bs8_lr1e2.pickle'
    
    # with open(path1, 'rb') as f:
    #     data1 = pickle.load(f)
    # result_dict['ZO-SGD, lr=1e-3, BS=32'] = average_across_batch(data1['Tr_Loss'], 1000).tolist()[0]
    
    # with open(path2, 'rb') as f:
    #     data2 = pickle.load(f)
    # result_dict['ZO-SGD, lr=1e-3, BS=16'] = average_across_batch(data2['Tr_Loss'], 1000).tolist()[0]
        
    # with open(path3, 'rb') as f:
    #     data3 = pickle.load(f)
    # result_dict['ZO-SGD, lr=1e-4, BS=8'] = average_across_batch(data3['Tr_Loss'], 1000).tolist()[0]
            
    # with open(path4, 'rb') as f:
    #     data4 = pickle.load(f)
    # result_dict['ZO-Coord, lr=4e-2, BS=32'] = average_across_batch(data4['Tr_Loss'], 1000).tolist()[0]
                
    # with open(path5, 'rb') as f:
    #     data5 = pickle.load(f)
    # result_dict['ZO-Coord, lr=1e-2, BS=16'] = average_across_batch(data5['Tr_Loss'], 1000).tolist()[0]
                    
    # with open(path6, 'rb') as f:
    #     data6 = pickle.load(f)
    # result_dict['ZO-Coord, lr=1e-2, BS=8'] = average_across_batch(data6['Tr_Loss'], 1000).tolist()[0]
    # #plot_results('LS with linear model (n=100, d=100):', result_dict, loss='Loss', lim_x=1000, lim_y=100)
    
    # result_dict = {}
    # path1 = 'results_LS_n1000_d100/LS_ZO_n1000_d100_bs32_lr1e3.pickle'
    # path2 = 'results_LS_n1000_d100/LS_ZO_n1000_d100_bs16_lr1e3.pickle'
    # path3 = 'results_LS_n1000_d100/LS_ZO_n1000_d100_bs8_lr1e4.pickle'
    # #path4 = 'results_LS_n1000_d100/LS_ZO_SVRG_Coord_Rand_FD_n1000_d100_bs32_lr5e3.pickle'
    # path5 = 'results_LS_n1000_d100/LS_ZO_SVRG_Coord_Rand_FD_n1000_d100_bs16_lr2e3.pickle'
    # path6 = 'results_LS_n1000_d100/LS_ZO_SVRG_Coord_Rand_FD_n1000_d100_bs8_lr1e4.pickle'
    
    # with open(path1, 'rb') as f:
    #     data1 = pickle.load(f)
    # result_dict['ZO-SGD, lr=1e-3, BS=32'] = average_across_batch(data1['Tr_Loss'], 1000).tolist()[0]
    
    # with open(path2, 'rb') as f:
    #     data2 = pickle.load(f)
    # result_dict['ZO-SGD, lr=1e-3, BS=16'] = average_across_batch(data2['Tr_Loss'], 1000).tolist()[0]
        
    # with open(path3, 'rb') as f:
    #     data3 = pickle.load(f)
    # result_dict['ZO-SGD, lr=1e-4, BS=8'] = average_across_batch(data3['Tr_Loss'], 1000).tolist()[0]
            
    # with open(path4, 'rb') as f:
    #     data4 = pickle.load(f)
    # result_dict['ZO-Coord, lr=5e-3, BS=32'] = average_across_batch(data4['Tr_Loss'], 1000).tolist()[0]
                
    # with open(path5, 'rb') as f:
    #     data5 = pickle.load(f)
    # result_dict['ZO-Coord, lr=2e-3, BS=16'] = average_across_batch(data5['Tr_Loss'], 1000).tolist()[0]
                    
    # with open(path6, 'rb') as f:
    #     data6 = pickle.load(f)
    # result_dict['ZO-Coord, lr=1e-4, BS=8'] = average_across_batch(data6['Tr_Loss'], 1000).tolist()[0]
    # #plot_results('LS with linear model (n=1000, d=100):', result_dict, loss='Loss', lim_x=1000, lim_y=100)

    # # MNIST
    # result_dict = {}
    # result_dict_zo = {}
    # result_dict_zo_svrg = {}
    # result_dict_step = {}
    # path1 = 'results_MNIST/MLP_ZO_bs64_lr2e3.pickle'
    # path2 = 'results_MNIST/MLP_ZO_bs32_lr1e3.pickle'
    # path3 = 'results_MNIST/MLP_ZO_bs16_lr1e3.pickle'
    # path4 = 'results_MNIST/MLP_ZO_SVRG_Coord_Rand_FD_bs64_lr1e2.pickle'
    # path5 = 'results_MNIST/MLP_ZO_SVRG_Coord_Rand_FD_bs32_lr1e2.pickle'
    # path6 = 'results_MNIST/MLP_ZO_SVRG_Coord_Rand_FD_bs16_lr1e3.pickle'
    
    # with open(path1, 'rb') as f:
    #     data1 = pickle.load(f)
    # result_dict['ZO-SGD, lr=2e-3, BS=64'] = average_across_batch(data1['Tr_Loss'], 10).tolist()[0]
    # result_dict_zo['ZO-SGD, lr=2e-3, BS=64'] = result_dict['ZO-SGD, lr=2e-3, BS=64']
    
    # with open(path2, 'rb') as f:
    #     data2 = pickle.load(f)
    # result_dict['ZO-SGD, lr=1e-3, BS=32'] = average_across_batch(data2['Tr_Loss'], 10).tolist()[0]
    # result_dict_zo['ZO-SGD, lr=1e-3, BS=32'] = result_dict['ZO-SGD, lr=1e-3, BS=32']
        
    # with open(path3, 'rb') as f:
    #     data3 = pickle.load(f)
    # result_dict['ZO-SGD, lr=1e-3, BS=16'] = average_across_batch(data3['Tr_Loss'], 10).tolist()[0]
    # result_dict_zo['ZO-SGD, lr=1e-3, BS=16'] = result_dict['ZO-SGD, lr=1e-3, BS=16']        
    
    # with open(path4, 'rb') as f:
    #     data4 = pickle.load(f)
    # result_dict_step['ZO-Coord, lr=1e-2, BS=64'] = data4['Tr_Loss']
    # result_dict['ZO-Coord, lr=1e-2, BS=64'] = average_across_batch(data4['Tr_Loss'], 10).tolist()[0]
    # result_dict_zo_svrg['ZO-Coord, lr=1e-2, BS=64'] = result_dict['ZO-Coord, lr=1e-2, BS=64']            
                
    # with open(path5, 'rb') as f:
    #     data5 = pickle.load(f)
    # result_dict_step['ZO-Coord, lr=1e-2, BS=32'] = data5['Tr_Loss']
    # result_dict['ZO-Coord, lr=1e-2, BS=32'] = average_across_batch(data5['Tr_Loss'], 10).tolist()[0]
    # result_dict_zo_svrg['ZO-Coord, lr=1e-2, BS=32'] = result_dict['ZO-Coord, lr=1e-2, BS=32']
                    
    # with open(path6, 'rb') as f:
    #     data6 = pickle.load(f)
    # result_dict_step['ZO-Coord, lr=1e-2, BS=16'] = data6['Tr_Loss']
    # result_dict['ZO-Coord, lr=1e-3, BS=16'] = average_across_batch(data6['Tr_Loss'], 10).tolist()[0]
    # result_dict_zo_svrg['ZO-Coord, lr=1e-3, BS=16'] = result_dict['ZO-Coord, lr=1e-3, BS=16']
    
    # #plot_results('MNIST with MLP (32-16)', result_dict, loss='Loss', lim_x=10, lim_y=3)
    
    # #plot_results_time('(ZO) MNIST with MLP (32-16): Time', result_dict_zo, [5, 7, 12,], loss='Loss', lim_x=100, lim_y=3)
    # #plot_results_time('(ZO-SVRG-Coord-Rand) MNIST with MLP (32-16): Time', result_dict_zo_svrg, [1320, 560, 400], loss='Loss', lim_x=10000, lim_y=3)

    # #plot_results_steps('MNIST with MLP (32-16): Steps', result_dict_step, loss='Loss', lim_x=500, lim_y=3)

    # # MNIST
    # result_dict_step = {}
    # path = 'results_MNIST/MLP_ZO_SVRG_Coord_Rand_FD150_bs64_lr1e3.pickle'

    # with open(path, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict_step['ZO-Coord, FD=150, lr=1e-2, BS=64'] = data['Tr_Loss']
    # result_dict_step['ZO-Coord, FD=40, lr=1e-2, BS=64'] = data4['Tr_Loss']
    # #plot_results_steps('MNIST with MLP (32-16): Steps', result_dict_step, loss='Loss', lim_x=1500, lim_y=3)


    # SVRG Experiments
    # LS with auxiliary only
    result_dict = {}
    result_dict_time = {}
    result_dict_query = {}
    
    path1 = 'results_SVRG/LS_ZO_SVRG_q1_n1000_d100_lr1e3_full.pickle'
    #path2 = 'results_SVRG/LS_ZO_n1000_d100_bs64_lr2e3.pickle'
    path3 = 'results_SVRG/LS_ZO_n1000_d100_bs32_lr1e3.pickle'
    # path4 = 'results_SVRG/LS_ZO_n1000_d100_bs16_lr1e3.pickle'
    #path5 = 'results_SVRG/LS_ZO_SVRG_q2_n1000_d100_lr1e3_bs32_full.pickle'
    #path6 = 'results_SVRG/LS_ZO_SVRG_q2_n1000_d100_lr1e3_bs64_full.pickle'
    #path8 = 'results_SVRG/LS_ZO_SVRG_Coord_Rand_FD_n1000_d100_bs64_lr2e3.pickle'
    path9 = 'results_SVRG/LS_SGD_n1000_d100_bs32_lr1e2.pickle'

    with open(path9, 'rb') as f:
        data = pickle.load(f)
    result_dict['FO-SGD'] = data['Tr_Loss']
    result_dict_time['FO-SGD'] = data['Time']
    result_dict_query['FO-SGD'] = data['Query']
    
    
    # with open(path2, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SGD, lr=2e-3, BS=64'] = data['Tr_Loss']
    # result_dict_time['ZO-SGD, lr=2e-3, BS=64'] = data['Time']
    # result_dict_query['ZO-SGD, lr=2e-3, BS=64'] = data['Query']

    with open(path3, 'rb') as f:
        data = pickle.load(f)
    result_dict['MeZO'] = data['Tr_Loss']
    result_dict_time['MeZO'] = data['Time']
    result_dict_query['MeZO'] = data['Query']


    with open(path1, 'rb') as f:
        data = pickle.load(f)
    result_dict['MeZO-SVRG'] = data['Tr_Loss']
    result_dict_time['MeZO-SVRG'] = data['Time']
    result_dict_query['MeZO-SVRG'] = data['Query']

    # with open(path5, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG,  lr=1e-4, BS=32, q=2'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG,  lr=1e-4, BS=32, q=2'] = data['Time']
    # result_dict_query['ZO-SVRG,  lr=1e-4, BS=32, q=2'] = data['Query']
    
    # with open(path6, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=1e-5, BS=64, q=2'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=1e-5, BS=64, q=2'] = data['Time']
    # result_dict_query['ZO-SVRG, lr=1e-5, BS=64, q=2'] = data['Query']
    
    # with open(path8, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG-C-R, lr=2e-3, BS=64'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG-C-R, lr=2e-3, BS=64'] = data['Time']
    # result_dict_query['ZO-SVRG-C-R, lr=2e-3, BS=64'] = data['Query']
    
    plot_results('Least-Squares Convergence Comparison', result_dict, loss='Loss', lim_x=15000, lim_y=0.00001)
    # plot_results_time('LS (n=1000, d=100): Time', result_dict, result_dict_time, loss='Loss', lim_x=10000, lim_y=150)
    # plot_results_query('LS (n=1000, d=100): Query Complexity', result_dict, result_dict_query, loss='Loss', lim_x=1e8, lim_y=150)

    #SVRG Experiments - MNIST
    result_dict = {}
    result_dict_time = {}
    result_dict_query = {}
    
    path1 = 'results_MNIST/MNIST_FO_lr1e3_bs64.pickle'
    path2 = 'results_MNIST/MNIST_ZO_lr1e3_bs64.pickle'
    path3 = 'results_MNIST/MNIST_ZO_lr1e3_bs32.pickle'
    path4 = 'results_MNIST/MNIST_ZO_SVRG_q1_lr2e3.pickle'
    path5 = 'results_MNIST/MNIST_ZO_SVRG_q2_bs32_lr1e5.pickle'
    path6 = 'results_MNIST/MNIST_ZO_SVRG_q2_bs64_lr1e4.pickle'
    path7 = 'results_MNIST/MNIST_ZO_lr1e3_bs128.pickle'
    
    
    with open(path1, 'rb') as f:
        data = pickle.load(f)
    result_dict['FO-SGD, BS=64'] = data['Tr_Loss']
    # result_dict_time['FO-SGD, lr=1e-3, BS=64'] = data['Time']
    # result_dict_query['FO-SGD, lr=1e-3, BS=64'] = data['Query']
    
    with open(path7, 'rb') as f:
        data = pickle.load(f)
    result_dict['MeZO, BS=128'] = data['Tr_Loss']
    # result_dict_time['ZO-SGD, lr=1e-3, BS=128'] = data['Time']
    # result_dict_query['ZO-SGD, lr=1e-3, BS=128'] = data['Query']
    
    
    with open(path2, 'rb') as f:
        data = pickle.load(f)
    result_dict['MeZO, BS=64'] = data['Tr_Loss']
    # result_dict_time['ZO-SGD, lr=1e-3, BS=64'] = data['Time']
    # result_dict_query['ZO-SGD, lr=1e-3, BS=64'] = data['Query']
    
    with open(path3, 'rb') as f:
        data = pickle.load(f)
    result_dict['MeZO, BS=32'] = data['Tr_Loss']
    # result_dict_time['ZO-SGD, lr=1e-3, BS=32'] = data['Time']
    # result_dict_query['ZO-SGD, lr=1e-3, BS=32'] = data['Query']
    
    with open(path4, 'rb') as f:
        data = pickle.load(f)
    result_dict['ZO-SVRG, lr=2e-3, q=1'] = data['Tr_Loss']
    result_dict_time['ZO-SVRG, lr=2e-3, q=1'] = data['Time']
    result_dict_query['ZO-SVRG, lr=2e-3, q=1'] = data['Query']
    
    with open(path5, 'rb') as f:
        data = pickle.load(f)
    result_dict['ZO-SVRG, lr=1e-4, BS=32, q=2'] = data['Tr_Loss']
    result_dict_time['ZO-SVRG, lr=1e-4, BS=32, q=2'] = data['Time']
    result_dict_query['ZO-SVRG, lr=1e-4, BS=32, q=2'] = data['Query']
    
    with open(path6, 'rb') as f:
        data = pickle.load(f)
    result_dict['ZO-SVRG, lr=1e-5, BS=64, q=2'] = data['Tr_Loss']
    result_dict_time['ZO-SVRG, lr=1e-5, BS=64, q=2'] = data['Time']
    result_dict_query['ZO-SVRG, lr=1e-5, BS=64, q=2'] = data['Query']
    
    
    # plot_results('MNIST: Iteration Complexity', result_dict, loss='Loss', lim_x=27000, lim_y=2.5)
    # plot_results_time('MNIST: Time Complexity', result_dict, result_dict_time, loss='Loss', lim_x=10000, lim_y=2.5)
    #plot_results_query('MNIST: Query Complexity', result_dict, result_dict_query, loss='Loss', lim_x=1e9, lim_y=2.5)

    del result_dict['ZO-SVRG, lr=2e-3, q=1']
    del result_dict['ZO-SVRG, lr=1e-4, BS=32, q=2']
    del result_dict['ZO-SVRG, lr=1e-5, BS=64, q=2']
    
    
    plot_results('MNIST Classification: Batch Size Experiment', result_dict, loss='Training Loss', lim_x=27000, lim_y=2.5)
    
    
    # MNLI Experiments
    # result_dict = {}
    # result_dict_time = {}
    # result_dict_query = {}
    
    # path1 = 'results_MNLI/MNLI_FO_lr1e2_bs32.pickle'
    # path2 = 'results_MNLI/MNLI_ZO_lr1e5_bs32.pickle'
    # path3 = 'results_MNLI/MNLI_ZO_lr1e4_bs64_withc.pickle'
    # path4 = 'results_MNLI/MNLI_ZO_SVRG_q1_lr5e4_wc_full.pickle'
    # path5 = 'results_MNLI/MNLI_ZO_SVRG_q1_lr1e4_wc_full.pickle'
    # path6 = 'results_MNLI/MNLI_ZO_SVRG_q1_lr1e3_withc_full.pickle'
    # path7 = 'results_MNLI/MNLI_ZO_SVRG_q1_lr1e4_withc_full.pickle'
    
    # with open(path1, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['FO-SGD, lr=1e-2, BS=32'] = data['Tr_Loss']
    # result_dict_time['FO-SGD, lr=1e-2, BS=32'] = data['Time']
    # result_dict_query['FO-SGD, lr=1e-2, BS=32'] = data['Query']
    
    # with open(path2, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SGD, lr=1e-5, BS=32'] = data['Tr_Loss']
    # result_dict_time['ZO-SGD, lr=1e-5, BS=32'] = data['Time']
    # result_dict_query['ZO-SGD, lr=1e-5, BS=32'] = data['Query']
    
    # with open(path3, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SGD, lr=1e-5, BS=64'] = data['Tr_Loss']
    # result_dict_time['ZO-SGD, lr=1e-5, BS=64'] = data['Time']
    # result_dict_query['ZO-SGD, lr=1e-5, BS=64'] = data['Query']
    
    # with open(path4, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=5e-4, q=1, no CLIP'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=5e-4, q=1, no CLIP'] = data['Time']
    # result_dict_query['ZO-SVRG, lr=5e-4, q=1, no CLIP'] = data['Query']
    
    # with open(path5, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=1e-4, q=1, no CLIP'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=1e-4, q=1, no CLIP'] = data['Time']
    # result_dict_query['ZO-SVRG, lr=1e-4, q=1, no CLIP'] = data['Query']

    # with open(path6, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=1e-3, q=1, CLIP'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=1e-3, q=1, CLIP'] = data['Time']
    # result_dict_query['ZO-SVRG, lr=1e-3, q=1, CLIP'] = data['Query']

    # with open(path7, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=1e-4, q=1, CLIP'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=1e-4, q=1, CLIP'] = data['Time']
    # result_dict_query['ZO-SVRG, lr=1e-4, q=1, CLIP'] = data['Query']

    # plot_results('MNLI: Iteration Complexity, mezo', result_dict, loss='Loss', lim_x=10000, lim_y=1.12)
    # plot_results_time('MNLI: Time Complexity, mezo', result_dict, result_dict_time, loss='Loss', lim_x=1e5, lim_y=1.12)
    # plot_results_query('MNLI: Query Complexity, mezo', result_dict, result_dict_query, loss='Loss', lim_x=1e8, lim_y=1.12)


    # MNLI Experiments - proj gradient
    # result_dict = {}
    # result_dict_time = {}
    # result_dict_query = {}
    # result_dict_norm = {}
    
    # path1 = 'results_MNLI_clipping/MNLI_ZO_SVRG_q1_lr5e4_withclip_maxnorm10_full.pickle'
    # path2 = 'results_MNLI_clipping/MNLI_ZO_SVRG_q1_lr5e4_withclip_maxnorm15_full.pickle'
    # path3 = 'results_MNLI_clipping/MNLI_ZO_SVRG_q1_lr5e4_withclip_maxnorm20_full.pickle'
    # path4 = 'results_MNLI_clipping/MNLI_ZO_SVRG_q1_lr5e4_noclip_full.pickle'
    
    # with open(path1, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=5e-4, maxnorm=1'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=5e-4, maxnorm=1'] = data['Time']
    # result_dict_query['ZO-SVRG, lr=5e-4, maxnorm=1'] = data['Query']
    # result_dict_norm['ZO-SVRG, lr=5e-4, maxnorm=1'] = data['Grad_Norm']
    
    # with open(path2, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=5e-4, maxnorm=1.5'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=5e-4, maxnorm=1.5'] = data['Time']
    # result_dict_query['ZO-SVRG, lr=5e-4, maxnorm=1.5'] = data['Query']
    # result_dict_norm['ZO-SVRG, lr=5e-4, maxnorm=1.5'] = data['Grad_Norm']
    
    # with open(path3, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=5e-4, maxnorm=2'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=5e-4, maxnorm=2'] = data['Time']
    # result_dict_query['ZO-SVRG, lr=5e-4, maxnorm=2'] = data['Query']
    # result_dict_norm['ZO-SVRG, lr=5e-4, maxnorm=2'] = data['Grad_Norm']

    # with open(path4, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=5e-4, no clip, warmup'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=5e-4, no clip, warmup'] = data['Time']
    # result_dict_query['ZO-SVRG, lr=5e-4, no clip, warmup'] = data['Query']
    # result_dict_norm['ZO-SVRG, lr=5e-4, no clip, warmup'] = data['Grad_Norm']

    
    # plot_results('MNLI: Iteration Complexity, norm', result_dict, loss='Loss', lim_x=10000, lim_y=3)
    # plot_norm('MNLI: Norm Evolution', result_dict, loss='Norm', lim_x=100, lim_y=2)
    
    
    # # MNLI Experiments - FO warmup
    # result_dict = {}
    # result_dict_time = {}
    # result_dict_query = {}
    # result_dict_norm = {}
    
    # path1 = 'results_MNLI/MNLI_FO_lr1e2_bs32.pickle'
    # path2 = 'results_MNLI_warmup/MNLI_ZO_SVRG_q1_lr5e4_noclip_full_warmup.pickle'
    # path3 = 'results_MNLI_warmup/MNLI_ZO_SVRG_q1_lr5e3_noclip_full_warmup.pickle'
    # path4 = 'results_MNLI_warmup/MNLI_ZO_SVRG_q1_lr1e4_noclip_full_warmup.pickle'
    # path5 = 'results_MNLI_warmup/MNLI_ZO_SVRG_q1_lr5e3_withclip_maxnorm20_full_warmup.pickle'

    
    # with open(path1, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['FO-SGD, lr=1e-2, BS=32, no clip'] = data['Tr_Loss']
    # result_dict_time['FO-SGD, lr=1e-2, BS=32, no clip'] = data['Time']
    # result_dict_query['FO-SGD, lr=1e-2, BS=32, no clip'] = data['Query']
    
    
    # with open(path2, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=5e-4, no clip'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=5e-4, no clip'] = data['Time']
    # result_dict_query['ZO-SVRG, lr=5e-4, no clip'] = data['Query']
    # result_dict_norm['ZO-SVRG, lr=5e-4, no clip'] = data['Grad_Norm']

    # with open(path3, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=5e-3, no clip'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=5e-3, no clip'] = data['Time']
    # result_dict_query['ZO-SVRG, lr=5e-3, no clip'] = data['Query']
    # result_dict_norm['ZO-SVRG, lr=5e-3, no clip'] = data['Grad_Norm']

    # with open(path4, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=1e-4, no clip'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=1e-4, no clip'] = data['Time']
    # result_dict_query['ZO-SVRG, lr=1e-4, no clip'] = data['Query']
    # result_dict_norm['ZO-SVRG, lr=1e-4, no clip'] = data['Grad_Norm']

    # with open(path5, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=5e-3, maxclip=2.0'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=5e-3, maxclip=2.0'] = data['Time']
    # result_dict_query['ZO-SVRG, lr=5e-3, maxclip=2.0'] = data['Query']
    # result_dict_norm['ZO-SVRG, lr=5e-3, maxclip=2.0'] = data['Grad_Norm']


    # plot_results('MNLI: Iteration Complexity, warmup', result_dict, loss='Loss', lim_x=500, lim_y=1.12)


    # MNLI Experiments - proj gradient
    # result_dict = {}
    # result_dict_time = {}
    # result_dict_query = {}
    # result_dict_norm = {}
    
    # path1 = 'MNLI_ZO_SVRG_q1_lr5e4_noclip_full_zstd1e0.pickle'
    # path2 = 'MNLI_ZO_SVRG_q1_lr5e4_noclip_full_zstd5e1.pickle'
    # path3 = 'MNLI_ZO_SVRG_q1_lr5e4_noclip_full_zstd1e1.pickle'
    # path4 = 'MNLI_ZO_SVRG_q1_lr5e4_noclip_full_zstd1e2.pickle'
    # path5 = 'MNLI_ZO_SVRG_q1_lr5e4_noclip_full_zstd1e3.pickle'

    
    # with open(path1, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-GD, lr=5e-4, z_std=1, no clip'] = data['Tr_Loss']
    # result_dict_time['ZO-GD, lr=5e-4, z_std=1, no clip'] = data['Time']
    # result_dict_query['ZO-GD, lr=5e-4, z_std=1, no clip'] = data['Query']
    # result_dict_norm['ZO-GD, lr=5e-4, z_std=1, no clip'] = data['Grad_Norm']

    # with open(path2, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-GD, lr=5e-4, z_std=5e-1, no clip'] = data['Tr_Loss']
    # result_dict_time['ZO-GD, lr=5e-4, z_std=5e-1, no clip'] = data['Time']
    # result_dict_query['ZO-GD, lr=5e-4, z_std=5e-1, no clip'] = data['Query']
    # result_dict_norm['ZO-GD, lr=5e-4, z_std=5e-1, no clip'] = data['Grad_Norm']

    # with open(path3, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-GD, lr=5e-4, z_std=1e-1, no clip'] = data['Tr_Loss']
    # result_dict_time['ZO-GD, lr=5e-4, z_std=1e-1, no clip'] = data['Time']
    # result_dict_query['ZO-GD, lr=5e-4, z_std=1e-1, no clip'] = data['Query']
    # result_dict_norm['ZO-GD, lr=5e-4, z_std=1e-1, no clip'] = data['Grad_Norm']

    # with open(path4, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-GD, lr=5e-4, z_std=1e-2, no clip'] = data['Tr_Loss']
    # result_dict_time['ZO-GD, lr=5e-4, z_std=1e-2, no clip'] = data['Time']
    # result_dict_query['ZO-GD, lr=5e-4, z_std=1e-2, no clip'] = data['Query']
    # result_dict_norm['ZO-GD, lr=5e-4, z_std=1e-2, no clip'] = data['Grad_Norm']


    # with open(path5, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-GD, lr=5e-4, z_std=1e-3, no clip'] = data['Tr_Loss']
    # result_dict_time['ZO-GD, lr=5e-4, z_std=1e-3, no clip'] = data['Time']
    # result_dict_query['ZO-GD, lr=5e-4, z_std=1e-3, no clip'] = data['Query']
    # result_dict_norm['ZO-GD, lr=5e-4, z_std=1e-3, no clip'] = data['Grad_Norm']


    # plot_norm('MNLI: Norm Evolution (l2-norm)', convert_to_cpu_float(result_dict_norm), loss='Norm', lim_x=10000, lim_y=2000)
    # plot_results('MNLI: Iteration Complexity,', result_dict, loss='Loss', lim_x=10000, lim_y=1.15)


    # # MNLI Experiments - proj gradient
    # result_dict = {}
    # result_dict_time = {}
    # result_dict_query = {}
    # result_dict_norm = {}
    
    # path1 = 'MNLI_ZO_SVRG_q1_lr5e4_noclip_full_zstd1e0.pickle'
    # path2 = 'MNLI_ZO_SVRG_q2_lr5e4_noclip_bs64_zstd1e0.pickle'

    
    # with open(path1, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-GD, lr=5e-4, z_std=1, no clip'] = data['Tr_Loss']
    # result_dict_time['ZO-GD, lr=5e-4, z_std=1, no clip'] = data['Time']
    # result_dict_query['ZO-GD, lr=5e-4, z_std=1, no clip'] = data['Query']
    # result_dict_norm['ZO-GD, lr=5e-4, z_std=1, no clip'] = data['Grad_Norm']
    
    # with open(path2, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=5e-4, z_std=1, no clip, q=2'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=5e-4, z_std=1, no clip, q=2'] = data['Time']
    # result_dict_query['ZO-SVRG, lr=5e-4, z_std=1, no clip, q=2'] = data['Query']
    # result_dict_norm['ZO-SVRG, lr=5e-4, z_std=1, no clip, q=2'] = data['Grad_Norm']

    # plot_results('MNLI: Iteration Complexity, with SVRG', result_dict, loss='Loss', lim_x=10000, lim_y=1.15)
    # plot_results_time('MNLI: Time Complexity, with SVRG', result_dict, result_dict_time, loss='Loss', lim_x=1e7, lim_y=1.12)

    # LR Adaptation
    # result_dict = {}
    # result_dict_time = {}
    # result_dict_query = {}
    # result_dict_norm = {}
    
    # path1 = 'MNLI_ZO_SVRG_q1_lr5e4_noclip_full_zstd1e0.pickle'
    # path2 = 'results_MNLI_LR_adaptation/MNLI_ZO_SVRG_q1_lr5e4_noclip_full_zstd1e0_anneal5.pickle'
    # path3 = 'results_MNLI_LR_adaptation/MNLI_ZO_SVRG_q1_lr5e4_noclip_full_zstd1e0_anneal.pickle'
    
    # with open(path1, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-GD, lr=5e-4'] = data['Tr_Loss']
    # result_dict_time['ZO-GD, lr=5e-4'] = data['Time']
    # result_dict_query['ZO-GD, lr=5e-4'] = data['Query']
    # result_dict_norm['ZO-GD, lr=5e-4'] = data['Grad_Norm']
    
    # with open(path2, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-GD, lr=5e-4, reduction_factor=5'] = data['Tr_Loss']
    # result_dict_time['ZO-GD, lr=5e-4, reduction_factor=5'] = data['Time']
    # result_dict_query['ZO-GD, lr=5e-4, reduction_factor=5'] = data['Query']
    # result_dict_norm['ZO-GD, lr=5e-4, reduction_factor=5'] = data['Grad_Norm']

    # with open(path3, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-GD, lr=5e-4, reduction_factor=10'] = data['Tr_Loss']
    # result_dict_time['ZO-GD, lr=5e-4, reduction_factor=10'] = data['Time']
    # result_dict_query['ZO-GD, lr=5e-4, reduction_factor=10'] = data['Query']
    # result_dict_norm['ZO-GD, lr=5e-4, reduction_factor=10'] = data['Grad_Norm']


    # plot_results('MNLI: LR Adaptation', result_dict, loss='Loss', lim_x=1000, lim_y=1.15)
    
    # # LR Adaptation + varying q
    # result_dict = {}
    # result_dict_time = {}
    # result_dict_proj = {}
    # result_dict_norm = {}
    
    # path1 = 'results_MNLI_LR_adaptation/MNLI_ZO_SVRG_q1_lr5e4_anneal5.pickle'
    # path2 = 'results_MNLI_LR_adaptation/MNLI_ZO_SVRG_q1_lr5e4_anneal10.pickle'
    # path3 = 'results_MNLI_LR_adaptation/MNLI_ZO_SVRG_q2_lr5e4_anneal5.pickle'
    # path4 = 'results_MNLI_LR_adaptation/MNLI_ZO_SVRG_q2_lr5e4_anneal10.pickle'
    # path5 = 'results_MNLI_LR_adaptation/MNLI_ZO_SVRG_q5_lr5e4_anneal5.pickle'
    # path6 = 'results_MNLI_LR_adaptation/MNLI_ZO_SVRG_q5_lr5e4_anneal10.pickle'
    # path7 = 'results_MNLI_LR_adaptation/MNLI_ZO_SVRG_q10_lr5e4_anneal5.pickle'
    # path8 = 'results_MNLI_LR_adaptation/MNLI_ZO_SVRG_q10_lr5e4_anneal10.pickle'
    
    
    # with open(path1, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, q=1, lr=5e-4, anneal=5'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, q=1, lr=5e-4, anneal=5'] = data['Time']
    # result_dict_proj['ZO-SVRG, q=1, lr=5e-4, anneal=5'] = data['Proj_Val']
    # result_dict_norm['ZO-SVRG, q=1, lr=5e-4, anneal=5'] = data['Grad_Norm']


    # with open(path2, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, q=1, lr=5e-4, anneal=10'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, q=1, lr=5e-4, anneal=10'] = data['Time']
    # result_dict_proj['ZO-SVRG, q=1, lr=5e-4, anneal=10'] = data['Proj_Val']
    # result_dict_norm['ZO-SVRG, q=1, lr=5e-4, anneal=10'] = data['Grad_Norm']
    

    # with open(path3, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, q=2, lr=5e-4, anneal=5'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, q=2, lr=5e-4, anneal=5'] = data['Time']
    # result_dict_proj['ZO-SVRG, q=2, lr=5e-4, anneal=5'] = data['Proj_Val']
    # result_dict_norm['ZO-SVRG, q=2, lr=5e-4, anneal=5'] = data['Grad_Norm']


    # with open(path4, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, q=2, lr=5e-4, anneal=10'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, q=2, lr=5e-4, anneal=10'] = data['Time']
    # result_dict_proj['ZO-SVRG, q=2, lr=5e-4, anneal=10'] = data['Proj_Val']
    # result_dict_norm['ZO-SVRG, q=2, lr=5e-4, anneal=10'] = data['Grad_Norm']
    

    # with open(path5, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, q=5, lr=5e-4, anneal=5'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, q=5, lr=5e-4, anneal=5'] = data['Time']
    # result_dict_proj['ZO-SVRG, q=5, lr=5e-4, anneal=5'] = data['Proj_Val']
    # result_dict_norm['ZO-SVRG, q=5, lr=5e-4, anneal=5'] = data['Grad_Norm']


    # with open(path6, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, q=5, lr=5e-4, anneal=10'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, q=5, lr=5e-4, anneal=10'] = data['Time']
    # result_dict_proj['ZO-SVRG, q=5, lr=5e-4, anneal=10'] = data['Proj_Val']
    # result_dict_norm['ZO-SVRG, q=5, lr=5e-4, anneal=10'] = data['Grad_Norm']
    
    # with open(path7, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, q=10, lr=5e-4, anneal=5'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, q=10, lr=5e-4, anneal=5'] = data['Time']
    # result_dict_proj['ZO-SVRG, q=10, lr=5e-4, anneal=5'] = data['Proj_Val']
    # result_dict_norm['ZO-SVRG, q=10, lr=5e-4, anneal=5'] = data['Grad_Norm']


    # with open(path8, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, q=10, lr=5e-4, anneal=10'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, q=10, lr=5e-4, anneal=10'] = data['Time']
    # result_dict_proj['ZO-SVRG, q=10, lr=5e-4, anneal=10'] = data['Proj_Val']
    # result_dict_norm['ZO-SVRG, q=10, lr=5e-4, anneal=10'] = data['Grad_Norm']


    # plot_results('MNLI: LR Adaptation + Varying q', result_dict, loss='Loss', lim_x=8000, lim_y=1.15)
    # plot_norm('MNLI: LR Adaptation + Varying q (l2-norm)', convert_to_cpu_float(result_dict_norm), loss='Norm', lim_x=8000, lim_y=7000)
    # plot_norm('MNLI: LR Adaptation + Varying q (Proj_grad values)', result_dict_proj, loss='Abs(Proj Values)', lim_x=8000, lim_y=7000)

    # LR Adaptation + varying q
    # result_dict = {}
    # result_dict_time = {}
    # result_dict_proj = {}
    # result_dict_norm = {}

    # path1 = 'MNLI_FO_lr1e3_bs64.pickle'
    # path2 = 'MNLI_ZO_lr1e5_bs64.pickle'
    # path3 = 'MNLI_ZO_SVRG_q2_lr5e4_anneal1.pickle'
    # path4 = 'MNLI_ZO_SVRG_q5_lr5e4_anneal1.pickle'
    # path5 = 'MNLI_ZO_SVRG_q10_lr5e4_anneal1.pickle'
    # path6 = 'MNLI_ZO_SVRG_q100_lr5e4_anneal1.pickle'   
   

    
    
    # with open(path1, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['FO-SGD, lr=1e-3, BS=64'] = data['Tr_Loss']
    # result_dict_time['FO-SGD, lr=1e-3, BS=64'] = data['Time']
    # #result_dict_proj['FO-SGD, lr=1e-3, BS=64'] = data['Proj_Val']
    # result_dict_norm['FO-SGD, lr=1e-3, BS=64'] = data['Grad_Norm']


    # with open(path2, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SGD, lr=1e-5, BS=64'] = data['Tr_Loss']
    # result_dict_time['ZO-SGD, lr=1e-5, BS=64'] = data['Time']
    # #result_dict_proj['ZO-SGD, lr=1e-5, BS=64'] = data['Proj_Val']
    # #result_dict_norm['ZO-SGD, lr=1e-5, BS=64'] = data['Grad_Norm']

    # with open(path3, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=5e-4, BS=64, q=2'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=5e-4, BS=64, q=2'] = data['Time']

    # with open(path4, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=5e-4, BS=64, q=5'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=5e-4, BS=64, q=5'] = data['Time']
    
    # with open(path5, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=5e-4, BS=64, q=10'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=5e-4, BS=64, q=10'] = data['Time']
    
    # with open(path6, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=5e-4, BS=64, q=100'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=5e-4, BS=64, q=100'] = data['Time']

    
    # plot_results('MNLI: Varying Q, Finetune Last 2 Layers', result_dict, loss='Loss', lim_x=16000, lim_y=1.15)

    ###########################

    # result_dict = {}
    # result_dict_time = {}
    # result_dict_proj = {}
    # result_dict_norm = {}

    # path1 = 'MNLI_FO_lr1e3_bs64.pickle'
    # path2 = 'MNLI_ZO_lr1e5_bs64.pickle'
    # path3 = 'MNLI_ZO_SVRG_q1_lr1e3_anneal1_eps1e3.pickle'
    # path4 = 'MNLI_ZO_SVRG_q2_lr1e3_anneal1_eps1e3.pickle'
    # path5 = 'MNLI_ZO_SVRG_q5_lr1e3_anneal1_eps1e3.pickle'
    # path6 = 'MNLI_ZO_SVRG_q100_lr1e3_anneal1_eps1e3.pickle'
    
    # with open(path1, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['FO-SGD, lr=1e-3, BS=64'] = data['Tr_Loss']
    # result_dict_time['FO-SGD, lr=1e-3, BS=64'] = data['Time']
    # #result_dict_proj['FO-SGD, lr=1e-3, BS=64'] = data['Proj_Val']
    # result_dict_norm['FO-SGD, lr=1e-3, BS=64'] = data['Grad_Norm']


    # with open(path2, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SGD, lr=1e-5, BS=64'] = data['Tr_Loss']
    # result_dict_time['ZO-SGD, lr=1e-5, BS=64'] = data['Time']
    # #result_dict_proj['ZO-SGD, lr=1e-5, BS=64'] = data['Proj_Val']
    # #result_dict_norm['ZO-SGD, lr=1e-5, BS=64'] = data['Grad_Norm']
    
    # with open(path3, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=1e-3, BS=64, q=1'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=1e-3, BS=64, q=1'] = data['Time']

    # with open(path4, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=1e-3, BS=64, q=2'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=1e-3, BS=64, q=2'] = data['Time']

    # with open(path5, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=1e-3, BS=64, q=5'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=1e-3, BS=64, q=5'] = data['Time']

    # with open(path6, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=1e-3, BS=64, q=100'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=1e-3, BS=64, q=100'] = data['Time']

    
    
    # plot_results('MNLI: Varying Q, Finetune Last 2 Layers, Larger LR', result_dict, loss='Loss', lim_x=9000, lim_y=1.15)
    # plot_results_time('MNLI: Time Complexity, Finetune Last 2 Layers, Larger LR', result_dict, result_dict_time, loss='Loss', lim_x=1e6, lim_y=1.12)


    # result_dict = {}
    # result_dict_time = {}
    # result_dict_proj = {}
    # result_dict_norm = {}

    # path1 = 'MNLI_ZO_SVRG_q1_lr5e4_anneal1_eps1e4.pickle'
    # path2 = 'MNLI_ZO_SVRG_q1_lr5e4_anneal1_eps1e5.pickle'
    # path3 = 'MNLI_ZO_SVRG_q2_lr1e3_anneal1_eps1e3.pickle'
    
    # with open(path1, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=5e-4, eps=1e-4'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=5e-4, eps=1e-4'] = data['Time']
    # #result_dict_proj['FO-SGD, lr=1e-3, BS=64'] = data['Proj_Val']
    # result_dict_norm['ZO-SVRG, lr=5e-4, eps=1e-4'] = data['Grad_Norm']

    # with open(path2, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=5e-4, eps=1e-5'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=5e-4, eps=1e-5'] = data['Time']
    # #result_dict_proj['FO-SGD, lr=1e-3, BS=64'] = data['Proj_Val']
    # result_dict_norm['ZO-SVRG, lr=5e-4, eps=1e-5'] = data['Grad_Norm']
    
    # with open(path3, 'rb') as f:
    #     data = pickle.load(f)
    # result_dict['ZO-SVRG, lr=5e-4, eps=1e-3'] = data['Tr_Loss']
    # result_dict_time['ZO-SVRG, lr=5e-4, eps=1e-3'] = data['Time']
    # #result_dict_proj['FO-SGD, lr=1e-3, BS=64'] = data['Proj_Val']
    # result_dict_norm['ZO-SVRG, lr=5e-4, eps=1e-3'] = data['Grad_Norm']
    
    # plot_norm('MNLI: l2-norm', convert_to_cpu_float(result_dict_norm), loss='Norm', lim_x=3000, lim_y=1000)


###########################

#     result_dict = {}
#     result_dict_time = {}
#     result_dict_proj = {}
#     result_dict_norm = {}

#     path1 = 'mnli_ZO_SVRG_q2_lr0.001_bs64_samplesize1024_fullparamTrue_anneal5_randpermute.pickle'
#     path2 = 'mnli_ZO_SVRG_q2_lr0.0005_bs64_samplesize1024_fullparamTrue.pickle'

    
#     with open(path1, 'rb') as f:
#         data = pickle.load(f)
#     result_dict['ZO-SVRG, randpermute'] = data['Tr_Loss']
#     result_dict_time['ZO-SVRG, lr=5e-4, BS=64, q=1'] = data['Time']

    
#     with open(path2, 'rb') as f:
#         data = pickle.load(f)
#     result_dict['ZO-SVRG, lr=5e-4, BS=64, q=2'] = data['Tr_Loss']
#     result_dict_time['ZO-SVRG, lr=5e-4, BS=64, q=2'] = data['Time']


#     plot_results('MNLI: Debug', result_dict, loss='Loss', lim_x=20000, lim_y=1.15)

# ###########################

#     result_dict = {}
#     result_dict_time = {}
#     result_dict_proj = {}
#     result_dict_norm = {}

#     path1 = 'MNLI_ZO_SVRG_q1_lr5e4_debug_shuffled.pickle'
#     path2 = 'MNLI_ZO_SVRG_q1_lr5e4_debug.pickle'

    
#     with open(path1, 'rb') as f:
#         data = pickle.load(f)
#     result_dict['ZO-GD, lr=5e-4, shuffle'] = data['Tr_Loss']

    
#     with open(path2, 'rb') as f:
#         data = pickle.load(f)
#     result_dict['ZO-GD, lr=5e-4, no shuffle'] = data['Tr_Loss']


#     plot_results('MNLI: q=1, Shuffle v No Shuffle', result_dict, loss='Loss', lim_x=5000, lim_y=1.15)


# ###########################

#     result_dict = {}
#     result_dict_time = {}
#     result_dict_proj = {}
#     result_dict_norm = {}

#     path1 = 'mnli_ZO_SVRG_q1_lr0.0005_bs64_samplesize1024_fullparamTrue.pickle'
#     path3 = 'mnli_ZO_SVRG_q2_lr0.0005_bs64_samplesize1024_fullparamTrue.pickle'
#     path5 = 'mnli_ZO_SVRG_q5_lr0.0005_bs64_samplesize1024_fullparamTrue.pickle'
#     path7 = 'mnli_ZO_SVRG_q10_lr0.0005_bs64_samplesize1024_fullparamTrue.pickle'
#     path2 = 'mnli_ZO_SVRG_q1_lr0.001_bs64_samplesize1024_fullparamTrue.pickle'
#     path4 = 'mnli_ZO_SVRG_q2_lr0.001_bs64_samplesize1024_fullparamTrue.pickle'
#     path6 = 'mnli_ZO_SVRG_q5_lr0.001_bs64_samplesize1024_fullparamTrue.pickle'
#     path8 = 'mnli_ZO_SVRG_q10_lr0.001_bs64_samplesize1024_fullparamTrue.pickle'

    
#     with open(path1, 'rb') as f:
#         data = pickle.load(f)
#     result_dict['ZO-SVRG, q=1, lr=5e-4'] = data['Tr_Loss']

    
#     with open(path2, 'rb') as f:
#         data = pickle.load(f)
#     result_dict['ZO-SVRG, q=1, lr=1e-3'] = data['Tr_Loss']

#     with open(path3, 'rb') as f:
#         data = pickle.load(f)
#     result_dict['ZO-SVRG, q=2, lr=5e-4'] = data['Tr_Loss']

    
#     with open(path4, 'rb') as f:
#         data = pickle.load(f)
#     result_dict['ZO-SVRG, q=2, lr=1e-3'] = data['Tr_Loss']
    
#     with open(path5, 'rb') as f:
#         data = pickle.load(f)
#     result_dict['ZO-SVRG, q=5, lr=5e-4'] = data['Tr_Loss']

    
#     with open(path6, 'rb') as f:
#         data = pickle.load(f)
#     result_dict['ZO-SVRG, q=5, lr=1e-3'] = data['Tr_Loss']

#     with open(path7, 'rb') as f:
#         data = pickle.load(f)
#     result_dict['ZO-SVRG, q=10, lr=5e-4'] = data['Tr_Loss']

    
#     with open(path8, 'rb') as f:
#         data = pickle.load(f)
#     result_dict['ZO-SVRG, q=10, lr=1e-3'] = data['Tr_Loss']



#     plot_results('MNLI: Random Sampling, Total Training Loss, Anneal=5', result_dict, loss='Loss', lim_x=10000, lim_y=1.15)


# ###########################

#     result_dict = {}
#     result_dict_time = {}
#     result_dict_proj = {}
#     result_dict_norm = {}

#     path1 = 'mnli_ZO_SVRG_q2_lr0.001_bs64_samplesize1024_fullparamTrue_anneal1.5.pickle'
#     path2 = 'mnli_ZO_SVRG_q2_lr0.001_bs64_samplesize1024_fullparamTrue_anneal2.pickle'
#     path3 = 'mnli_ZO_SVRG_q2_lr0.001_bs64_samplesize1024_fullparamTrue_anneal3.pickle'
#     path4 = 'mnli_ZO_SVRG_q2_lr0.001_bs64_samplesize1024_fullparamTrue_anneal4.pickle'
#     path5 = 'mnli_ZO_SVRG_q2_lr0.001_bs64_samplesize1024_fullparamTrue.pickle'

    
#     with open(path1, 'rb') as f:
#         data = pickle.load(f)
#     result_dict['ZO-SVRG, q=2, anneal=1.5'] = data['Tr_Loss']

#     with open(path2, 'rb') as f:
#         data = pickle.load(f)
#     result_dict['ZO-SVRG, q=2, anneal=2'] = data['Tr_Loss']

#     with open(path3, 'rb') as f:
#         data = pickle.load(f)
#     result_dict['ZO-SVRG, q=2, anneal=3'] = data['Tr_Loss']
    
#     with open(path4, 'rb') as f:
#         data = pickle.load(f)
#     result_dict['ZO-SVRG, q=2, anneal=4'] = data['Tr_Loss']

#     with open(path5, 'rb') as f:
#         data = pickle.load(f)
#     result_dict['ZO-SVRG, q=2, anneal=5'] = data['Tr_Loss']


#     plot_results('MNLI: Effect of Annealing Factor', result_dict, loss='Loss', lim_x=10000, lim_y=1.15)
