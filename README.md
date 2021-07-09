# Python programming code for the bachelor thesis of Marco Deken, Vrije Universiteit Amsterdam, 9 July 2021
Simulating and Estimating (Grouped) Panel Data

This repository is split up into 6 main files with functionality:
    #1: simulate.py
        This file holds the class Simulate, with which it is possible to simulate panel data with
        different forms of fixed effects, different forms of slope parameters, or different forms
        of errors.

    #2: Bon_Man_GFE.py
        This file holds the code of a class that implements the Grouped Fixed Effects estimator as described
        by Bonhomme and Manresa, 2015. Also a main() is included, to give an example of how to use the class
        and how to estimate parameters of a dataset simulated with simulate.py.
    #3: Lin_Ng_CKmeans.py
        This file holds the code of a class that implements the Conditional K-means algorithm, as described
        by Lin and Ng, 2012. Also a main() is included, to give an example of how to use the class
        and how to estimate parameters of a dataset simulated with simulate.py.
    #4: Lin_Ng_PSEUDO.py
        This file holds the code of a class that implements the Two-step pseudo threshold algorithm as
        described by Lin and Ng, 2012. Also a main() is included, to give an example of how to use the class
        and how to estimate parameters of a dataset simulated with simulate.py.

    #5: new_estimate.py
        This file was used to run all of the simulation experiments that have been done for my Bachelor Thesis.
        It lets all three of the algorithms (GFE, CK-means, Pseudo) estimate on simulated datasets of different
        sizes of N and T, or different data generating processes in general (i.e. heteroskedastic errors). It
        saves the results in a Result class, also defined in this python file. The results are saved in separate
        folders for each of the three algorithms, with filenames that described the context in which the model
        was estimated (i.e. number of individuals N, number of time-points T, number of iterations M, the type
        of errors, etc.). Finally, with the help of TrackTime it keeps track of the time how long everything
        takes to estimate. The results of these computation times are saved in a dictionary (with the filenames
        as keys) in a separate file: comp_times_dict.py.

    #6: plot.py
        This file holds all the code that was used to make plots of the results obtained in new_simulate.py.
