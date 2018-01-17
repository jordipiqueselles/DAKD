import os
import pandas as pd
import scipy.stats as st


if __name__ == '__main__':
    folder = "./raw_results/"
    df = pd.DataFrame()
    classifier = []
    dataset = []
    halfPca = []
    bestPca = []
    bestFeatures = []
    nComp = []
    for file in os.listdir(folder):
        with open(folder + file, 'r') as f:
            for line in f:
                if "Accuracy half pca feat:" in line:
                    accuracy = float(line.split()[-1])
                    halfPca.append(accuracy)
                elif "Accuracy best pca feat:" in line:
                    accuracy = float(line.split()[-1])
                    bestPca.append(accuracy)
                elif "nComp:" in line:
                    comp = int(line.split()[-1])
                    nComp.append(comp)
                    clss = file.split('_')[0]
                    classifier.append(clss)
                    dat = file.split('_')[1].split('.')[0]
                    dataset.append(dat)
                elif "Best features:" in line:
                    feat = line.replace("Best features:", "")
                    feat = feat.replace('\n', "")
                    feat = feat.replace(" ", "")
                    bestFeatures.append(feat)
    df['classifier'] = classifier
    df['dataset'] = dataset
    df['nComp'] = nComp
    df['bestFeat'] = bestFeatures
    df['halfPca'] = halfPca
    df['bestPca'] = bestPca
    df['difference'] = df['bestPca'] - df['halfPca']
    df.to_csv("results.csv", index=False)


    # Anova dataset
    if len(set(df.dataset)) > 1:
        dSetDf = pd.DataFrame()
        for dat in set(df.dataset):
            dSetDf[dat] = df[df.dataset == dat].difference.values
        anovaDataset = st.f_oneway(dSetDf.iloc[:,0], dSetDf.iloc[:,1], dSetDf.iloc[:,2], dSetDf.iloc[:,3])
        print("Anova dataset")
        print(anovaDataset)
        print()

    # Anova classifier
    if len(set(df.classifier)) > 1:
        clssDf = pd.DataFrame()
        for clss in set(df.classifier):
            clssDf[clss] = df[df.classifier == clss].difference.values
        anovaClss = st.f_oneway(clssDf.iloc[:,0], clssDf.iloc[:,1], clssDf.iloc[:,2], clssDf.iloc[:,3], clssDf.iloc[:,4])
        print("Anova classifier")
        print(anovaClss)
        print()