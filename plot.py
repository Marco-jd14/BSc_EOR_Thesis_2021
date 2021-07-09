# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:51:45 2021

@author: Marco
"""

import pandas as pd
import numpy as np
import os.path
import sys
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from simulate import Effects, Slopes, Variance, Dataset, Fit
from lib.tracktime import TrackTime, TrackReport
from Bon_Man import GFE
from Lin_Ng_PSEUDO import PSEUDO
from Lin_Ng_CKmeans import CK_means
from new_estimate import Result
from comp_times_dicts import computation_times


def plot(matrix, N_range, T_range, title, select, subtitle=""):
    y_min = np.min(matrix)
    y_max = np.max(matrix)
    delta = (y_max - y_min) * 0.05
    for s in range(len(select)):
        for n in range(len(N_range)):
            plt.plot(matrix[s,n,:], '-o', markersize=4)

        plt.title(title + model_names[select[s]].upper() + subtitle)
        plt.legend(["N=%d"%n for n in N_range])
        my_xticks = ["T=%d"%t for t in T_range]
        plt.xticks(range(len(T_range)), my_xticks)
        plt.ylim(y_min-delta, y_max+delta)
        plt.grid()
        plt.savefig("plots/%s"%title + model_names[select[s]].upper(), bbox_inches='tight', pad_inches=0.05, dpi=250)
        plt.show()

def calc_min_max(list_of_df):
    x_min = np.Inf
    x_max = -np.Inf
    for s in range(len(list_of_df)):
        for n in range(len(list_of_df[s])):
            for t in range(len(list_of_df[s][n])):
                temp_min = np.min(list_of_df[s][n][t]['Lower CI'])
                temp_max = np.max(list_of_df[s][n][t]['Upper CI'])
                x_min = temp_min if temp_min < x_min else x_min
                x_max = temp_max if temp_max > x_max else x_max

    return x_min, x_max


def plot_CI(list_of_df, N_range, T_range, slopes_true, title, select, subtitle=""):
    x_min, x_max = calc_min_max(list_of_df)
    x_min = max(x_min,-0.6)
    x_max = min(x_max,1.2)
    delta = (x_max - x_min) * 0.05
    nr_exps = len(N_range) * len(T_range)
    for s in range(len(select)):
        for n in range(len(N_range)):
            for t in range(len(T_range)):

                sorted_means = np.sort(list_of_df[s][n][t]['Mean'])
                nr_slopes = len(list_of_df[s][n][t])
                colours = ["#000000","#696969"]
                for counter in range(nr_slopes):
                    slope = np.where(sorted_means[counter] == list_of_df[s][n][t]['Mean'])[0]
                    lower_ci = list_of_df[s][n][t]['Lower CI'][slope]
                    upper_ci = list_of_df[s][n][t]['Upper CI'][slope]
                    y_coordinate = (n*len(T_range)+t)*2+(counter-int(nr_slopes/2))*0.22
                    CI, = plt.plot((lower_ci, upper_ci), [y_coordinate]*2, 'o-', color=colours[counter%2], markersize=2.5)
                    red_dot, = plt.plot(list_of_df[s][n][t]['Mean'][slope], y_coordinate, 'ro', markersize=4)

        slope = slopes_true.values.flatten()[0]
        for slope in slopes_true.values.flatten():
            slope_line, = plt.plot((slope,slope),(-int(nr_slopes/2)*0.22,(nr_exps-1)*2+int(nr_slopes/2)*0.22), 'b--', alpha=0.7, linewidth=1)

        plt.legend([slope_line, red_dot, CI], ['True slope value', 'Avg estimate', 'Conf. Int.'], loc="upper left", ncol=3)

        my_yticks = ["N=%d T=%d"%(N,T) for N in N_range for T in T_range]
        plt.yticks(np.arange(nr_exps)*2, my_yticks)
        plt.xlim(x_min-delta, x_max+delta)
        bottom, top = plt.ylim()
        plt.ylim(bottom, top+1.5)
        plt.title(title + model_names[select[s]].upper() + subtitle)
        plt.savefig("plots/%s"%title + model_names[select[s]].upper(), bbox_inches='tight', pad_inches=0.05, dpi=250)
        plt.show()


def plot_G(matrix, N_range, T_range, true_G, title, select, subtitle=""):

    for s in range(len(select)):
        fig, axs = plt.subplots(nrows=len(N_range),ncols=1,figsize=(6.5,10))
        for n in range(len(N_range)):
            legend = {"red": [0,"Ĝ<G°"] , "green": [0,"Ĝ=G°"], "royalblue": [0,"Ĝ>G°"]}
            averages = np.zeros(len(T_range))
            for t in range(len(T_range)):
                averages[t] = np.average(matrix[s,n,t,:])
                unique, counts = np.unique(matrix[s,n,t,:], return_counts=True)
                counts = counts/np.sum(counts)

                for i in range(len(unique)):
                    x1 = unique[i]-0.5
                    x2 = unique[i]+0.5
                    y = counts[i] + t
                    axs[n].plot((y,y), (x1,x2), "#303030",linewidth=1)
                    axs[n].plot((t,y), (x1,x1), "#303030",linewidth=1)
                    axs[n].plot((t,y), (x2,x2), "#303030",linewidth=1)
                    axs[n].plot((t,t), (x1,x2), "#303030",linewidth=1)
                    axs[n].set_ylabel("Ĝ",rotation=0,labelpad=10)
                    if unique[i] == true_G:
                        color = "green"
                    elif unique[i] < true_G:
                        color = "red"
                    else:
                        color = "royalblue"
                    if legend[color][0]:
                        axs[n].fill_between(x=(t,y), y1=(x1,x1), y2=(x2,x2), color=color)
                    else:
                        axs[n].fill_between(x=(t,y), y1=(x1,x1), y2=(x2,x2), color=color, label=legend[color][1])
                        legend[color][0] = 1

            axs[n].plot(averages,"-o",label="average Ĝ",color="k",linewidth=3,markerfacecolor="white",markeredgewidth=2)

            axs[n].legend(ncol=2)
            axs[n].set_ylim(0,true_G+4+1)
            axs[n].set_xlim(-0.25,len(T_range) + 0.25)
            my_xticks = ["T=%d"%t for t in T_range] + [""]
            axs[n].set_xticks(range(len(T_range)+1))
            axs[n].set_xticklabels(my_xticks)
            axs[n].grid(linestyle='--', color='gray')
            axs[n].set_title("N=%d"%N_range[n],size=11)

        plt.suptitle(title + model_names[select[s]].upper() + subtitle,y=0.94,size=14)
        plt.subplots_adjust(hspace=0.3)
        plt.savefig("plots/%s"%title + model_names[select[s]].upper(), bbox_inches='tight', pad_inches=0.05, dpi=250)
        plt.show()

def calc_tot_dict_time():
    tot_time_sec = sum(computation_times.values())
    hours = int(tot_time_sec/60/60)
    tot_time_sec -= hours*60*60
    minutes = int(tot_time_sec/60)
    seconds = int(tot_time_sec - minutes*60)
    print("Calculating everything in the dictionary took %d hours, %d minutes, and %d seconds\n"%(hours,minutes,seconds))

def tot_comp_time(comp_times):
    tot_time_sec = 0.0
    for a in range(len(comp_times)):
        for b in range(len(comp_times[a])):
            for c in range(len(comp_times[a][b])):
                tot_time_sec += comp_times[a][b][c]
    hours = int(tot_time_sec/60/60)
    tot_time_sec -= hours*60*60
    minutes = int(tot_time_sec/60)
    seconds = int(tot_time_sec - minutes*60)
    print("Calculating this all took %d hours, %d minutes, and %d seconds\n"%(hours,minutes,seconds))

# model_names = ["gfe (gr-tvar-fix)", "ckmeans (ind-fix)", "pseudo (ind-fix)", "gfe (ind-fix)", "ckmeans (gr-tvar-fix)", "pseudo (gr-tvar-fix)"]
# model_names = ["gfe (g_max=2)", "ckmeans (g_max=2)", "pseudo (g_max=2)", "gfe (g_min=6)", "ckmeans (g_min=6)", "pseudo (g_min=6)"]
# model_names = ["CKMEANS (heteroskedasticity)", "CKMEANS (3 DoF)", "CKMEANS (10 DoF)"]
model_names = ["gfe (g_max=8)", "ckmeans (g_max=8)", "gfe (g_max=2)", "ckmeans (g_max=2)", "gfe (g_min=6)", "ckmeans (g_min=6)"]
# model_names = ["gfe", "ckmeans", "pseudo"]
def main():
    select = [0,1,2,3,4,5]

    G = 4
    K = 1
    M = 100
    fit = Fit.complete
    var = Variance.homosk
    DoF = 0

    T_range = [5, 10, 20, 50, 100]
    T_range_CI = T_range#[10, 50, 100]
    N_range = [50, 100, 200]
    N_range_CI = [50, 200]

    RMSE_beta  = np.zeros((len(select), len(N_range), len(T_range)))
    RMSE_alpha = np.zeros((len(select), len(N_range), len(T_range)))
    comp_times = np.zeros((len(select), len(N_range), len(T_range)))
    accuracy   = np.zeros((len(select), len(N_range), len(T_range)))
    G_hats     = np.zeros((len(select), len(N_range), len(T_range)))
    G_hats2    = np.zeros((len(select), len(N_range), len(T_range), M), dtype=int)
    conf_interval = [[[[] for i in T_range_CI] for j in N_range_CI] for s in select]

    for s in range(len(select)):
        model_name = model_names[select[s]].split(" ")[0].lower()
        for n in range(len(N_range)):
            for t in range(len(T_range)):
                N = N_range[n]
                T = T_range[t]
                eff = (Effects.ind_fix if select[s]==3 else Effects.gr_tvar_fix)
                filename = "%s/%s_N=%d_T=%d_G=%d_K=%d_M=%d_fit=%d_e=%d" %(model_name,model_name,N,T,G,K,M,fit.value,var.value)
                # filename += "_eff=%d"%eff.value if select[s] >= 3 else ""
                # filename += "_dof=%d"%DoF if DoF > 0 else ""
                filename += "_gmax=2" if model_names[select[s]][-9:] == "(g_max=2)" else ""
                filename += "_gmin=6" if model_names[select[s]][-9:] == "(g_min=6)" else ""
                # if s == 0:
                #     filename += "_e=1"
                # elif s == 1:
                #     filename += "_e=0_dof=3"
                # else:
                #     filename += "_e=0_dof=10"
                print(filename)
                if os.path.isfile(filename):
                    with open(filename, 'rb') as output:
                        result = pickle.load(output)
                else:
                    print("INCOMPLETE DATA: '%s'\n"%filename)
                    sys.exit(1)

                comp_times[s,n,t] = computation_times[filename[len(model_name)+1:]]

                RMSE_beta[s,n,t] = result.RMSE()*100
                RMSE_alpha[s,n,t] = result.RMSE_alpha*100

                result.confusion_mat_groups()
                accuracy[s,n,t] = np.trace(result.conf_mat) / np.sum(result.conf_mat.values)

                if T in T_range_CI and N in N_range_CI:
                    result.conf_interval(0.05)
                    conf_interval[s][N_range_CI.index(N)][T_range_CI.index(T)] = result.summary

                if fit == Fit.complete:
                    subtitle = " (G° unknown)"
                    G_hats[s,n,t] = len(result.G_hats[result.G_hats==result.slopes_true.shape[1]]) / len(result.G_hats)
                    G_hats2[s,n,t,:] = result.G_hats[:]

    if fit == Fit.complete:
        subtitle = " (G° unknown)"
    elif var == Variance.heterosk:
        subtitle = " (heteroskedasticity)"
    elif DoF == 0:
        subtitle = ""
    else:
        subtitle = " (%d DoF)"%DoF
    subtitle = "" if len(model_names) == 6 else subtitle

    tot_comp_time(comp_times)
    plot(comp_times, N_range, T_range, "Computation times (sec) - ", select, subtitle)
    plot(RMSE_beta, N_range, T_range, "slope RMSE - ", select, subtitle)
    plot(RMSE_alpha, N_range, T_range, "fixed effects RMSE - ", select, subtitle)
    if fit == Fit.complete:
        # plot(G_hats, N_range, T_range, "G estimation accuracy - ", select, subtitle)
        plot_G(G_hats2, N_range, T_range, result.slopes_true.shape[1], "Estimating G - ", select, subtitle)
    else:
        plot(accuracy, N_range, T_range, "Group classification accuracy - ", select, subtitle)
        plot_CI(conf_interval, N_range_CI, T_range_CI, result.slopes_true, "Confidence Intervals - ", select, subtitle)


if __name__ == "__main__":
    main()