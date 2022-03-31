# Machine Learning in Bioinformatics (Master's thesis)

This project is about the  analysis of bioinformatics data, and it has been realised for during my master's thesis writing.


## Machine Learning and Deep Learning: an application in bioinformatics

### Abstract:
In the big data era, to transform biomedical data in useful knowledge is one of the most important challenges in bioinformatics. 
In order to understand the structure of a cell and its functioning, the valuable information needed concerns the amount of mRNA 
produced by each gene of the cells DNA (also referred to as gene expression). Thanks to microarray technologies it is possible to 
consider simultaneously, in only one experiment, up to 30 thousands genes on each cell-line, thus gathering a huge amount of data.

When it comes to evaluate the eﬃcacy of a new drug, the experiment consists in measuring the gene expression of a tumor cell-line 
before the drug is administered and then using it to predict its response to the treatment. The analysis presented in this thesis 
is based on data about 18916 genes disposed on 464 cell-lines of diﬀerent tumors (13 diﬀerent types of cancer) collected by several
experiments of the Mario Negri Instituite for Pharmacological Research. The eﬃcacy of each treatment is evaluated on the basis of 
two quantitative variables, AUC and IC50, representing respectively the area under the curve of the dose-response plot and the 
maximal concentration of drug to cause 50% inhibition of biological activity of cancer cell.From a statistical point of view, 
given the numeric nature of the variables of interest, the problem can be described as a regression model in which each gene 
expression acts as a predictor for the drug eﬃcacy. However, the huge amount of genes and the high costs of each records detection 
generate many practical complications, so that is not possible to apply classical regression methods without a proper pre-elaboration
of the data. As far as this thesis is concerned, a computationally eﬃcient method using parallel computing and both Python and R
optimized libraries is dealt with in order to assess the relationship between gene-expression and drug response in the framework
described above.

Firstly the responses to drugs in the 13 Cancer type cell-lines are compared by means of both a Kruskal-Wallis test and a 
multiple paired WilcoxonMann-Whitney test using Bonferroni correction. Moreover, given the enormous number of predictors, 
a Principal Component Analysis is performed before applying the Machine Learning algorithm in order to reduce the problems 
dimensionality without losing information. As a result the ﬁrst 300 components are kept, covering more than 95% of the total
explained variance.

Secondly, two Machine Learning methods (Linear Regression and Support Vector Machine) are adopted to estimate the drug response 
using PCA components as predictors. In particular, the independence of the drug response from the cancer type is investigated in
ﬁrst place using as training set all the cell-line types. In second place, the drug response of every single cell-line type is 
predicted from all the other cancer types. Lastly, blood cell-lines are used as baseline predictors for estimating drug response
of each kind of tumor. Validation and Testing are then conducted using k-fold cross validation in order to exploit all available
information in each step of analysis.

Thirdly, in order to capture the non-linear relationship between from gene-expression and drug response - suggested by non-linear 
choice of the kernel in the validation phase of the SVM - a Deep learning algorithm, namely the Multilayer feed forward neural network,
is also explored using several conﬁgurations in validation step.
Finally, the results of the Machine Learning and the Deep Learning approaches are compared. Additionally, after a brief discussion 
about some possible alternatives to optimize the computational eﬀort is dealt with, the entire analysis is repeated using the CINECA’s
supercomputer MARCONI and exploiting the advantages Graphics Processing Unit (GPU) parallel computing instead of the classical 
multithreading parallelism.

### Maintainer
Antonio Macaluso
