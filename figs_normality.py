import os
from itertools import product
from time import time

import numpy as np 
from scipy.stats import norm,shapiro,anderson,kstest,wasserstein_distance

import matplotlib.pyplot as plt 
from read_h5 import read_h5 

def normality_plot(dist="uniform",
                   loads=[0.125, 0.15, 0.175],
                   temps=[0.05,0.1,0.15],
                   ks = [1],
                   fibers=[1,2,3,4,5,6,7,8,9,10,
                           11,12,13,14,15,16,17,18,19,20,
                           30,40,50,60,70,80,90,100,200,400,600,800,
                           1000,2000,4000,6000,8000,10000,20000,100000], 
                   h5file="/FASTTEMP/shiemer/fiber-bundles-safe.h5"):
    """
    Plot Fig. 4 and 5. in paper. 
    
    
    In the first run, the test statistics for all 
    combinations of the parameters loads,temps,ks and fibers are calculated 
    from raw data found in the HDF5 file which are then stored in files 
    "comb-comparison_"+dist+".csv", "mean-comparison_"+dist+".csv", 
    "var-comparison_"+dist+".csv". In subsequent runs it looks for these files
    and checks whether the calculations have already been done and looks up the 
    result. The idea here however is that in the first run, all combinations 
    for a given distribution have been already calculated. So do not expect 
    this program to run without errors if you have added new data to the HDF5 
    file since the last time. If in doubt delete "comb-comparison_"+dist+".csv", 
    "mean-comparison_"+dist+".csv", "var-comparison_"+dist+".csv" always gives 
    correct results.

    Parameters
    ----------
    dist : str 
        name of distribution. Either "uniform" or "weibull".
    loads : list
        load per fiber (f0 in paper).
    temps : list
        temperature (symbol T in paper).
    ks : list
        distribution parameter (symbol T in paper). For weibull the exponent,
        for uniform the span.
    fibers : list
        number of fibers. 
    h5file : str
        hdf5 file in which to find the raw data.

    Returns
    -------
    None

    """
    colors = plt.get_cmap("tab10")
    font=28

    if os.path.isfile("comb-normality_"+dist+".csv") and \
       os.path.isfile("normality_"+dist+".csv"):

        # load already calculated values
        combs = np.loadtxt("comb-normality_"+dist+".csv")
        statistics = np.loadtxt("normality_"+dist+".csv")
        #
        assert np.allclose(combs, np.loadtxt("comb-comparison_"+dist+".csv"))
        #
        mean = np.loadtxt("mean-comparison_"+dist+".csv")
        var = np.loadtxt("var-comparison_"+dist+".csv")
        #
        _combs = list(product(loads,temps,ks,fibers))
        mask,indices = find_matching_rows(A=np.array(_combs),
                                          B=combs)
        #
        _statistics = []
        ind = 0
        for load,temp,k,fiber in _combs:
            # check if done already
            done_already = mask[ind].all()
            # if combination done already, just take old result
            if done_already:
                i = indices[ind]
                _statistics.append(statistics[i])
                _mean.append(mean[i])
                _var.append(var[i])

            else:
                print("Calculate: ", load,temp,k,fiber)
                timeseries,n_fibers,aval = read_h5(fibers=fiber,
                                                   h5file=h5file,
                                                   load=load,
                                                   temperature=temp,
                                                   k=1,
                                                   distribution=dist,
                                                   subset=None)

                # extract lifetimes
                lifetimes = np.array([t[0][-1] for t in timeseries])
                #
                _mean.append(np.mean(lifetimes))
                _var.append(np.var(lifetimes))
                #
                andson = anderson(lifetimes,
                                  dist="norm",
                                  method="interpolate")
                andson = tuple([andson.statistic, andson.pvalue])
                #
                kolmog = kstest(rvs=lifetimes,
                                cdf="norm",
                                args=(mean[-1], var[-1]),
                method = "exact",
                alternative = "two-sided",
                nan_policy = "omit")
                kolmog = tuple([kolmog.statistic, kolmog.pvalue])
                #
                _statistics.append(shapiro(x=lifetimes,
                                           nan_policy="omit") + \
                                   andson + \
                                   kolmog)
            #
            ind += 1

        statistics = _statistics
        combs = np.array(combs).astype(float)
        statistics = np.array(statistics)
    else:
        combs = list(product(loads,temps,ks,fibers))
        statistics = []
        mean = []
        var = []
        for load,temp,k,fiber in combs:
            #
            timeseries,n_fibers,aval = read_h5(fibers=fiber,
                                               h5file=h5file,
                                               load=load,
                                               temperature=temp,
                                               k=1,
                                               distribution=dist,
                                               subset=None)
            # extract lifetimes
            lifetimes = np.array([t[0][-1] for t in timeseries])
            #
            mean.append(np.mean(lifetimes))
            var.append(np.var(lifetimes))
            #
            t = time()
            andson = anderson(lifetimes,
                              dist="norm",
                              method="interpolate")
            andson = tuple([andson.statistic, andson.pvalue])
            print("anderson: ",time()-t)
            #
            kolmog = kstest(rvs=lifetimes,
                            cdf="norm",
                            args=(mean[-1],np.sqrt(var[-1])),
                            method="exact",
                            alternative="two-sided",
                            nan_policy="omit")
            kolmog = tuple([kolmog.statistic,kolmog.pvalue])
            print("Kolmogorov: ", time() - t)
            #
            statistics.append(shapiro(x=lifetimes,
                                       nan_policy="omit")+\
                               andson+\
                               kolmog+\
                               tuple([wasserstein_distance(lifetimes,
                                        norm.rvs(loc=mean[-1], scale=np.sqrt(var[-1]),
                                                 size=len(lifetimes)))]))
            print("Shapiro: ", time() - t)
        #
        mean, var = np.array(mean), np.array(var)
        #
        combs = np.array(combs).astype(float)

        #
        statistics = np.array(statistics)
        #
        np.savetxt("comb-normality_"+dist+".csv",combs,
                   header="load,temp,k,fiber")
        np.savetxt("normality_"+dist+".csv", statistics,
                   header="shapiro-wilkinson statistic, shapiro-wilkinson p-value, "+\
                          "anderson statistic, anderson p-value, "+\
                          "kolmogorov-smirnov statistic, kolmogorov-smirnov p-value, "+\
                          "empirical wasserstein distance", )
    #
    N = np.arange(1,np.max(fibers)+1)
    #
    _loads,_temps,_ks,_fibers = combs[:,0],combs[:,1],combs[:,2],combs[:,3]
    # create two fig
    fig,axs = plt.subplots(1,2,figsize=(15, 10))

    # each temperature-load combination gets its own color
    i = 0
    for load,temp,k in product(loads,temps,ks):

        # create mask
        mask = (_temps == temp) & (_loads==load)

        # plot shapiro
        axs[0].scatter(_fibers[mask],statistics[mask,-2],
                       color=colors(i),
                       label=str(temp)+", "+str(np.round(load,3)))
        axs[1].scatter(_fibers[mask],statistics[mask,-2],
                       color=colors(i),
                       label=str(temp)+", "+str(np.round(load,3)))
        # plot anderson
        axs[0].scatter(_fibers[mask],statistics[mask,-2],
                       color=colors(i),
                       label=str(temp)+", "+str(np.round(load,3)))
        axs[1].scatter(_fibers[mask],statistics[mask,-2],
                       color=colors(i),
                       label=str(temp)+", "+str(np.round(load,3)))
        i += 1
    #
    axs[0].set_xlim(left=np.min(fibers)*0.9,
                    right=np.max(fibers)*3)
    axs[0].set_ylim(bottom=1e-30,
                    top=1.)
    axs[1].set_xlim(left=np.min(fibers)*0.9,
                    right=np.max(fibers)*3)
    axs[1].set_ylim(bottom=1e-3,
                    top=1.)
    #
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    #
    axs[0].set_yticks(np.logspace(-30,0,31))
    axs[1].set_yticks(np.logspace(-3,0,4))
    #
    left = np.floor( np.log10(fibers).min() )
    right = np.ceil( np.log10(fibers).max() )
    xticks = np.logspace(left, right, int(right-left + 1) )
    axs[0].set_xticks( xticks )
    axs[1].set_xticks( xticks )
    #
    axs[0].set_xlabel("N",fontsize=font)
    axs[1].set_xlabel("N",fontsize=font)
    axs[0].set_ylabel(r"$p value$",fontsize=font)
    axs[1].set_ylabel(r"$p value$",fontsize=font)


    axs[1].legend(loc='center left',
         bbox_to_anchor=(1, 0.5),
         fontsize=font,
         title=r"T , $f_{0}$",
         title_fontsize=font,
         frameon=False,
         borderpad=0)

    axs[0].tick_params(axis='both', which='both', labelsize=font)
    axs[1].tick_params(axis='both', which='both', labelsize=font)

    plt.subplots_adjust(wspace=0.35)
    plt.savefig("normality-comparison_"+dist+".pdf",
                format="pdf",
                bbox_inches="tight")

    return

def find_matching_rows(A, B):
    """
    Row wise equivalent to np.isin for a 2D numpy array. It generates a mask 
    (tells you if all entries of a row in array A are in array B) and 
    an index array in which row of B you find the row of A. This function is 
    not completely fool proof.

    Parameters
    ----------
    A : np.ndarray, shape (k,n)
        first array.
    B : np.ndarray, shape (m,n) or
        second array.

    Returns
    -------
    mask : np.ndarray, shape (k,n)
        if all entries of a row i are True, then row A[i] is also in B .
    indices : np.ndarray, shape (k,n)
        in which row to find A[i]
    """
    # use structured arrays that each row is like a single entry
    A_struct = A.view([('', A.dtype)] * A.shape[1])
    B_struct = B.view([('', B.dtype)] * B.shape[1])
    # mask for rows in A that also are in B
    mask = np.isin(A_struct, B_struct)
    # index in B for each True entry of mask
    indices = [ (B==row).all(axis=1).nonzero()[0][0] for row in A[mask[:,0]]]
    #
    return mask, np.array(indices)

if __name__ == "__main__":
    normality_plot(dist="uniform",
                   loads=[0.125, 0.15, 0.175],
                   fibers=[10000,20000,100000])
    normality_plot(dist="weibull",
                loads=np.array([0.5, 0.6, 0.7])*np.exp(-1),
                fibers=[
                        10000,20000,100000])
