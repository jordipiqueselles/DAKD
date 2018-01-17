# Improving the use of PCA

This program improves the use of PCA by applying a feature selection 
process (wrapper) to the components obtained by PCA

Here there are the instructions to execute the program
````commandline

python3 main.py folderDatasets minC maxC predictors [-v] [-p] [-h]
 
folderDatasets -> the folder that contains the datasets in csv format
minC -> Minimum number of components
maxC -> Maximum number of components
predictors -> List of predictors to use. The valid values are lr, nb, kn, dt, rf, all
-v (optional) -> verbose
-p (optional) -> parallel
-h (optional) -> help
 
Examples:
python3 main.py ./datasets 4 10 all -p
python3 main.py ./datasets 12 13 [lr,dt] -v
````

If you want to postprocess the results and obtain a csv file, you can
execute the **postProcess.py** script.

````commandline
python3 postProcess.py
````

It will read the files in the folder ./raw_results and will create 
the file results.csv