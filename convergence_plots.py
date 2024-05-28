import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Ensure that fonts are not converted to outlines in PDF
matplotlib.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.size': 12})

def moving_average(a, n=12):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_results(title, d, loss="Train Loss: ", lim_x=1000, lim_y=3):
    plt.figure()
    for k, val in d.items():

        v = np.array([val])
        means, mins, maxes = np.mean(v, axis=0), np.amin(v, axis=0), np.amax(v, axis=0)
        means = moving_average(means)
        plt.plot(range(1, len(means)+1), means, linewidth=2, linestyle='solid', markersize=12, label=k)
        #plt.fill_between(range(len(val['Tr_Loss'])), mins, maxes, alpha=0.5)
        
    plt.title(title)
    plt.yscale('log')
    plt.xlim(1, lim_x)
    plt.ylim(0.9, lim_y)
    plt.ylabel(loss)
    plt.xlabel('Step')
    plt.legend(bbox_to_anchor=(0.65, 1.0), loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.savefig(title + '.pdf')

def plot_results_time(title, d_y, d_x, fo_value, loss="Train Loss: ", lim_x_l=1000, lim_x=1000, lim_y=3):
    plt.figure()
    for k, val in d_y.items():
        v = np.array([val])
        v_x = np.linspace(0, d_x[k], num=v.size)
        
        means, mins, maxes = np.mean(v, axis=0), np.amin(v, axis=0), np.amax(v, axis=0)
        means = moving_average(means*100, n=8)
        l = len(means)
        plt.plot(v_x[:l], means, linewidth=2, linestyle='solid', markersize=12, label=k)
        #plt.fill_between(range(len(val['Tr_Loss'])), mins, maxes, alpha=0.5)
    plt.axhline(y=fo_value, color='black', linestyle='--', linewidth=2, label='FO-SGD')

    plt.title(title)
    plt.xscale('log')
    plt.xlim(lim_x_l, lim_x)
    plt.ylim(30, lim_y)
    plt.ylabel(loss)
    plt.xlabel('Time (s)')
    plt.legend(bbox_to_anchor=(0.0, 1), loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.savefig(title + '.pdf')
    
def plot_results_query(title, d_y, d_x, fo_value, loss="Train Loss: ", lim_x_l=1000, lim_x=1000, lim_y=3):
    plt.figure()
    for k, val in d_y.items():
        v = np.array([val])
        v_x = np.cumsum(np.array([d_x[k]]))
        
        means, mins, maxes = np.mean(v, axis=0), np.amin(v, axis=0), np.amax(v, axis=0)
        means = moving_average(means)
        l = len(means)
        plt.plot(v_x[:l], means, linewidth=2, linestyle='solid', markersize=12, label=k)
    plt.axhline(y=fo_value, color='black', linestyle='--', linewidth=2, label='FO-SGD')
    
    plt.title(title)
    #plt.yscale('log')
    plt.xscale('log')
    plt.xlim(lim_x_l, lim_x)
    plt.ylim(0, lim_y)
    plt.ylabel(loss)
    plt.xlabel('Queries')
    plt.legend(bbox_to_anchor=(0.0, 1), loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.savefig(title + '.pdf')

if __name__ == "__main__":
    # distilbert
    result_dict = {}
    result_dict_acc = {}
    result_dict_query = {}
    result_dict_time = {}

    path2 = 'distilbert/result_SST2_DistilBert_FullParam/distilbert-base-cased_sst2_ZO_SVRG_q2_lr0.001_bs64_samplesize512_fullparamTrue_anneal4.0_perturbationscale0.001.pickle'
    path1 = '/home/tgautamx/ZO_SmallScaleExp/distilbert/result_SST2_DistilBert_FullParam/distilbert-base-cased_sst2_ZO_lr1e-06_bs64_samplesize512_fullparamTrue_perturbationscale0.001.pickle'
    #path3 = 'mnli_ZO_SVRG_q2_lr0.001_bs64_samplesize1024_fullparamTrue_anneal3.pickle'
    FO_benchmark = 0.0012
    FO_benchmark_acc = 88

    with open(path1, 'rb') as f:
        data = pickle.load(f)
        result_dict['MeZO'] = data['Tr_Loss']
        result_dict_query['MeZO'] = data['Query']
        result_dict_time['MeZO'] = data['Overall_Tr_Time']
        result_dict_acc['MeZO'] = data['Val_Acc']
        
    with open(path2, 'rb') as f:
        data = pickle.load(f)
        result_dict['MeZO-SVRG'] = data['Tr_Loss']
        result_dict_query['MeZO-SVRG'] = data['Query']
        result_dict_time['MeZO-SVRG'] = data['Overall_Tr_Time']
        result_dict_acc['MeZO-SVRG'] = data['Val_Acc']
        
    plot_results_query('Fine-tuning DistilBert on SST-2: Query Plot', result_dict, result_dict_query, FO_benchmark, loss='Fine-tuning Loss', lim_y=1.0, lim_x=23e6, lim_x_l=1e5)
    plot_results_time('Fine-tuning DistilBert on SST-2: Time Plot', result_dict_acc, result_dict_time, FO_benchmark_acc, loss='Test Accuracy (%)', lim_y=105, lim_x=1e4)


    # roberta-large
    result_dict = {}
    result_dict_acc = {}
    result_dict_query = {}
    result_dict_time = {}

    path2 = '/home/tgautamx/ZO_SmallScaleExp/robertalarge/result_SST2_Robertalarge_FullParam/roberta-large_sst2_ZO_SVRG_q2_lr5e-05_bs64_samplesize512_fullparamTrue_anneal5.0_perturbationscale0.001.pickle'
    path1 = '/home/tgautamx/ZO_SmallScaleExp/robertalarge/result_SST2_Robertalarge_PartialParam/roberta-large_sst2_ZO_lr1e-06_bs64_samplesize512_fullparamFalse.pickle'
    #path3 = 'mnli_ZO_SVRG_q2_lr0.001_bs64_samplesize1024_fullparamTrue_anneal3.pickle'
    FO_benchmark = 0.108
    FO_benchmark_acc = 96

    with open(path1, 'rb') as f:
        data = pickle.load(f)
        result_dict['MeZO'] = data['Tr_Loss']
        result_dict_query['MeZO'] = data['Query']
        result_dict_time['MeZO'] = data['Overall_Tr_Time']
        result_dict_acc['MeZO'] = data['Val_Acc']
        
    with open(path2, 'rb') as f:
        data = pickle.load(f)
        result_dict['MeZO-SVRG'] = data['Tr_Loss']
        result_dict_query['MeZO-SVRG'] = data['Query']
        result_dict_time['MeZO-SVRG'] = data['Overall_Tr_Time']
        result_dict_acc['MeZO-SVRG'] = data['Val_Acc']
        
    plot_results_query('Fine-tuning RoBERTa-large on SST-2: Query Plot', result_dict, result_dict_query, FO_benchmark, loss='Fine-tuning Loss', lim_x_l=5e6, lim_y=1.0, lim_x=12.3e6)
    plot_results_time('Fine-tuning RoBERTa-large on SST-2: Time Plot', result_dict_acc, result_dict_time, FO_benchmark_acc, loss='Test Accuracy (%)', lim_y=125, lim_x=1.1e5, lim_x_l=5e4)
    
    # opt
    
    # qnli 
    result_dict = {}
    result_dict_acc = {}
    result_dict_query = {}
    result_dict_time = {}

    path2 = '/home/tgautamx/ZO_SmallScaleExp/opt/result_QNLI_OPT_FullParam/opt-2.7b_qnli_ZO_SVRG_q2_lr5e-05_bs64_samplesize512_fullparamTrue_anneal4.0_perturbationscale0.001.pickle'
    path1 = '/home/tgautamx/ZO_SmallScaleExp/opt/result_QNLI_OPT_FullParam/opt-2.7b_qnli_ZO_lr1e-07_bs64_samplesize512_fullparamTrue_perturbationscale0.001.pickle'
    #path3 = 'mnli_ZO_SVRG_q2_lr0.001_bs64_samplesize1024_fullparamTrue_anneal3.pickle'
    FO_benchmark = 0.12
    FO_benchmark_acc = 91

    with open(path1, 'rb') as f:
        data = pickle.load(f)
        result_dict['MeZO'] = data['Tr_Loss']
        result_dict_query['MeZO'] = data['Query']
        result_dict_time['MeZO'] = data['Overall_Tr_Time']
        result_dict_acc['MeZO'] = data['Val_Acc']
        
    with open(path2, 'rb') as f:
        data = pickle.load(f)
        result_dict['MeZO-SVRG'] = data['Tr_Loss']
        result_dict_query['MeZO-SVRG'] = data['Query']
        result_dict_time['MeZO-SVRG'] = data['Overall_Tr_Time']
        result_dict_acc['MeZO-SVRG'] = data['Val_Acc']
        
    plot_results_query('Fine-tuning OPT-2.7B on QNLI: Query Plot', result_dict, result_dict_query, FO_benchmark, loss='Fine-tuning Loss', lim_x_l=1e3, lim_y=1.4, lim_x=4096e3)
    plot_results_time('Fine-tuning OPT-2.7B on QNLI: Time Plot', result_dict_acc, result_dict_time, FO_benchmark_acc, loss='Test Accuracy (%)', lim_y=115, lim_x=2.6e5, lim_x_l=1e4)
    
    # mnli 
    result_dict = {}
    result_dict_acc = {}
    result_dict_query = {}
    result_dict_time = {}

    path2 = '/home/tgautamx/ZO_SmallScaleExp/opt/result_MNLI_OPT_FullParam/opt-2.7b_mnli_ZO_SVRG_q2_lr5e-05_bs64_samplesize512_fullparamTrue_anneal4.0_perturbationscale0.001.pickle'
    path1 = '/home/tgautamx/ZO_SmallScaleExp/opt/result_MNLI_OPT_FullParam/opt-2.7b_mnli_ZO_lr1e-07_bs64_samplesize512_fullparamTrue_perturbationscale0.001.pickle'
    #path3 = 'mnli_ZO_SVRG_q2_lr0.001_bs64_samplesize1024_fullparamTrue_anneal3.pickle'
    FO_benchmark = 0.12
    FO_benchmark_acc = 91

    with open(path1, 'rb') as f:
        data = pickle.load(f)
        result_dict['MeZO'] = data['Tr_Loss']
        result_dict_query['MeZO'] = data['Query']
        result_dict_time['MeZO'] = data['Overall_Tr_Time']
        result_dict_acc['MeZO'] = data['Val_Acc']
        
    with open(path2, 'rb') as f:
        data = pickle.load(f)
        result_dict['MeZO-SVRG'] = data['Tr_Loss']
        result_dict_query['MeZO-SVRG'] = data['Query']
        result_dict_time['MeZO-SVRG'] = data['Overall_Tr_Time']
        result_dict_acc['MeZO-SVRG'] = data['Val_Acc']
        
    plot_results_query('Fine-tuning OPT-2.7B on MNLI: Query Plot', result_dict, result_dict_query, FO_benchmark, loss='Fine-tuning Loss', lim_x_l=1e3, lim_y=1.4, lim_x=4096e3)
    plot_results_time('Fine-tuning OPT-2.7B on MNLI: Time Plot', result_dict_acc, result_dict_time, FO_benchmark_acc, loss='Test Accuracy (%)', lim_y=105, lim_x=2.6e5, lim_x_l=1e1)
    
    # sst2 
    result_dict = {}
    result_dict_acc = {}
    result_dict_query = {}
    result_dict_time = {}

    path2 = '/home/tgautamx/ZO_SmallScaleExp/opt/result_SST2_OPT_FullParam/opt-2.7b_sst2_ZO_SVRG_q2_lr5e-05_bs64_samplesize512_fullparamTrue_anneal4.0_perturbationscale0.001.pickle'
    path1 = '/home/tgautamx/ZO_SmallScaleExp/opt/result_SST2_OPT_FullParam/opt-2.7b_sst2_ZO_lr1e-07_bs64_samplesize512_fullparamTrue_perturbationscale0.001.pickle'
    #path3 = 'mnli_ZO_SVRG_q2_lr0.001_bs64_samplesize1024_fullparamTrue_anneal3.pickle'
    FO_benchmark = 0.12
    FO_benchmark_acc = 91

    with open(path1, 'rb') as f:
        data = pickle.load(f)
        result_dict['MeZO'] = data['Tr_Loss']
        result_dict_query['MeZO'] = data['Query']
        result_dict_time['MeZO'] = data['Overall_Tr_Time']
        result_dict_acc['MeZO'] = data['Val_Acc']
        
    with open(path2, 'rb') as f:
        data = pickle.load(f)
        result_dict['MeZO-SVRG'] = data['Tr_Loss']
        result_dict_query['MeZO-SVRG'] = data['Query']
        result_dict_time['MeZO-SVRG'] = data['Overall_Tr_Time']
        result_dict_acc['MeZO-SVRG'] = data['Val_Acc']
        
    plot_results_query('Fine-tuning OPT-2.7B on MNLI: Query Plot', result_dict, result_dict_query, FO_benchmark, loss='Fine-tuning Loss', lim_x_l=1e3, lim_y=1.4, lim_x=4096e3)
    plot_results_time('Fine-tuning OPT-2.7B on MNLI: Time Plot', result_dict_acc, result_dict_time, FO_benchmark_acc, loss='Test Accuracy (%)', lim_y=105, lim_x=2.6e5, lim_x_l=1e1)
    

    # sst2 
    result_dict = {}
    result_dict_acc = {}
    result_dict_query = {}
    result_dict_time = {}

    path2 = '/home/tgautamx/ZO_SmallScaleExp/opt/result_Cola_OPT_FullParam/opt-2.7b_cola_ZO_SVRG_q2_lr5e-05_bs64_samplesize512_fullparamTrue_anneal4.0_perturbationscale0.001.pickle'
    path1 = '/home/tgautamx/ZO_SmallScaleExp/opt/result_Cola_OPT_FullParam/opt-2.7b_cola_ZO_lr1e-07_bs64_samplesize512_fullparamTrue_perturbationscale0.001.pickle'
    #path3 = 'mnli_ZO_SVRG_q2_lr0.001_bs64_samplesize1024_fullparamTrue_anneal3.pickle'
    FO_benchmark = 0.12
    FO_benchmark_acc = 91

    with open(path1, 'rb') as f:
        data = pickle.load(f)
        result_dict['MeZO'] = data['Tr_Loss']
        result_dict_query['MeZO'] = data['Query']
        result_dict_time['MeZO'] = data['Overall_Tr_Time']
        result_dict_acc['MeZO'] = data['Val_Acc']
        
    with open(path2, 'rb') as f:
        data = pickle.load(f)
        result_dict['MeZO-SVRG'] = data['Tr_Loss']
        result_dict_query['MeZO-SVRG'] = data['Query']
        result_dict_time['MeZO-SVRG'] = data['Overall_Tr_Time']
        result_dict_acc['MeZO-SVRG'] = data['Val_Acc']
        
    plot_results_query('Fine-tuning OPT-2.7B on MNLI: Query Plot', result_dict, result_dict_query, FO_benchmark, loss='Fine-tuning Loss', lim_x_l=1e3, lim_y=1.4, lim_x=4096e3)
    plot_results_time('Fine-tuning OPT-2.7B on MNLI: Time Plot', result_dict_acc, result_dict_time, FO_benchmark_acc, loss='Test Accuracy (%)', lim_y=105, lim_x=2.6e5, lim_x_l=1e1)
    
    # gpt2 
    # qnli
    result_dict = {}
    result_dict_acc = {}
    result_dict_query = {}
    result_dict_time = {}

    path2 = '/home/tgautamx/ZO_SmallScaleExp/gpt2-xl/result_Cola_GPT2_FullParam/gpt2-xl_cola_ZO_SVRG_q2_lr7e-05_bs64_samplesize512_fullparamTrue_anneal4.0_perturbationscale0.001.pickle'
    path1 = '/home/tgautamx/ZO_SmallScaleExp/gpt2/result_Cola_GPT2_FullParam/gpt2-xl_cola_ZO_lr1e-06_bs64_samplesize512_fullparamTrue.pickle'
    #path3 = 'mnli_ZO_SVRG_q2_lr0.001_bs64_samplesize1024_fullparamTrue_anneal3.pickle'
    FO_benchmark = 0.48
    FO_benchmark_acc = 72
    
    with open(path1, 'rb') as f:
        data = pickle.load(f)
        result_dict['MeZO'] = data['Tr_Loss']
        result_dict_query['MeZO'] = data['Query']
        result_dict_time['MeZO'] = data['Overall_Tr_Time']
        result_dict_acc['MeZO'] = data['Val_Acc']
        
    with open(path2, 'rb') as f:
        data = pickle.load(f)
        result_dict['MeZO-SVRG'] = data['Tr_Loss']
        result_dict_query['MeZO-SVRG'] = data['Query']
        result_dict_time['MeZO-SVRG'] = data['Overall_Tr_Time']
        result_dict_acc['MeZO-SVRG'] = data['Val_Acc']
        
    plot_results_query('Fine-tuning GPT2-XL on MNLI: Query Plot', result_dict, result_dict_query, FO_benchmark, loss='Fine-tuning Loss', lim_x_l=1e5, lim_y=1.4, lim_x=4096e3)
    plot_results_time('Fine-tuning GPT2-XL on MNLI: Time Plot', result_dict_acc, result_dict_time, FO_benchmark_acc, loss='Test Accuracy (%)', lim_y=105, lim_x=2.6e5, lim_x_l=1e1)
    
    # sst2
    result_dict = {}
    result_dict_acc = {}
    result_dict_query = {}
    result_dict_time = {}

    path2 = '/home/tgautamx/ZO_SmallScaleExp/gpt2-xl/result_SST2_GPT2_FullParam/gpt2-xl_sst2_ZO_SVRG_q2_lr7e-05_bs64_samplesize512_fullparamTrue_anneal4.0_perturbationscale0.001.pickle'
    path1 = '/home/tgautamx/ZO_SmallScaleExp/gpt2/result_SST2_GPT2_FullParam/gpt2-xl_sst2_ZO_lr1e-06_bs64_samplesize512_fullparamTrue.pickle'
    #path3 = 'mnli_ZO_SVRG_q2_lr0.001_bs64_samplesize1024_fullparamTrue_anneal3.pickle'
    FO_benchmark = 0.27
    FO_benchmark_acc = 72
    
    with open(path1, 'rb') as f:
        data = pickle.load(f)
        result_dict['MeZO'] = data['Tr_Loss']
        result_dict_query['MeZO'] = data['Query']
        result_dict_time['MeZO'] = data['Overall_Tr_Time']
        result_dict_acc['MeZO'] = data['Val_Acc']
        
    with open(path2, 'rb') as f:
        data = pickle.load(f)
        result_dict['MeZO-SVRG'] = data['Tr_Loss']
        result_dict_query['MeZO-SVRG'] = data['Query']
        result_dict_time['MeZO-SVRG'] = data['Overall_Tr_Time']
        result_dict_acc['MeZO-SVRG'] = data['Val_Acc']
        
    plot_results_query('Fine-tuning GPT2-XL on QNLI: Query Plot', result_dict, result_dict_query, FO_benchmark, loss='Fine-tuning Loss', lim_x_l=1e5, lim_y=0.95, lim_x=3900e3)
    plot_results_time('Fine-tuning GPT2-XL on QNLI: Time Plot', result_dict_acc, result_dict_time, FO_benchmark_acc, loss='Test Accuracy (%)', lim_y=105, lim_x=1e5, lim_x_l=1e3)
    