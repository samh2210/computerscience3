This code focuses on solving the duplicate detection problem. 
This is to see when the same product is on different web shops. 
In this python file we first extract the data including all the duplicates.
Then there are 4 important functions written. 

First function is Minhash
The first function does locally sensitive hashing, it does all the steps needed in this function.
You need to fill in the sample you want to use and then the pairs that should be compared according to LSH will come out.
After this a few similarity functions will be written down, these are important for the second function which I am about to name.

Second functions is HSMalgorithm
In the second function the HSMalgorithm with some slight adjustments is done. 
Inputs are sample, alpha, beta, gamma, delta and place

Here you fill in the sample of your code, this is b_samp for the bootstrap and o_samp for the test sample. b_samp_00, b_samp_11 are in the first lines these can be filled in too
You then fill in all the paramaters. These are alpha, beta, gamma, delta
Lastly this function needs the place input. This is a matrix with the original positions of the bootstrap or test sample. For b_samp this is sidx, for o_samp this is ooidx. 
In the first 20 cells this place matrix is also determined for the other bootstrap and test samples.
Combinations are 
b_samp_11  loc11
b_samp_22  loc22
b_samp_33  loc33
b_samp  sidx
o_samp ooidx
o_samp0 oidx00
o_samp1 oidx11
o_samp2 oidx22
o_samp3 oidx33
The first function is in the second function, so by running the second function with these inputs the f1 score, pair quality and pair completeness will return. The outcome of the function is the f1 score the pair quality and completeness will be printed.

Third function is GridsearchHSM
The third function does a gridsearch over the hsmalgorithm. You fill in the sample over which you would like to do the gridsearch.

Fourth function is f1scorefortest
The fourth function runs the hsmalgorithm with the optimized parameters on the trainingset. You need to fill in o_samp and for the 5 test samples the hsmalgorithm will be running. 
