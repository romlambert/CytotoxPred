# toxifate.py
# use python 3.10

import numpy as np
import pandas as pd
import pycytominer as pcm # pip install git+https://github.com/cytomining/pycytominer
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def __init__(self, path):
    self.path = path
    
def extract_metadata_from_image_name(string="BR3-blasts1-r01c07f04p01.npy", format = 'RepPlateImage'): # WORK IN PROGRESS 17/11/2023
    if "." in string: #handle the case where the file extension is included
        string = string.split('.')[0]
    if len(string.split("-")) == 3: #handle the case where the replicate is included
        replicate = string.split('-')[0]
        cell_type = string.split('-')[1][:-1]
        plate_layout = string.split('-')[1][-1]
        well_id = string.split('-')[2]
        return replicate, cell_type, plate_layout, well_id
    elif len(string.split("-")) == 2: #handle the case where the replicate is not included
        cell_type = string.split('-')[0][:-1]
        plate_layout = string.split('-')[0][-1]
        well_id = string.split('-')[1]
        return cell_type, plate_layout, well_id
    elif len(string.split("-")) == 1: #handle the case where the replicate and cell type are not included
        plate_layout = string.split('-')[0][-1]
        well_id = string.split('-')[0][:-1] 
        
        return plate_layout, well_id


def PlateMap(i):
    switcher={
        1:['NCAP','ATOR','CERI','CLOF','COLC','DAPT'],
        2:['DEXA','DOXO','EZET','HYCQ','LEFL','SELU'],
        3:['CLEV','ETOP','FIAL','GEMF','IBIP','MCPP'],
        4:['ETHF','ETRE','NELA','SIMV','VORI','WURS'],
        5:['CISP','IMAT','OLAN','SUNI','TEBU','ZIDO']}
    return switcher.get(i,"Plate number not in 1-5")

def dataframe_scaler(df):
    # scale the data from a pandas dataframe, and return a new dataframe with same index and column names
    # needs a dataframe with numeric values only
    scaler = StandardScaler()
    scaler.fit(df)
    scaled_data = scaler.transform(df)
    # add the indices and column names back to the scaled data
    scaled_data = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
    return scaled_data

def FormatHarmony(path):
    # format single-cell harmony files by adding concentration, drug names, and renaming features
    conc = {'Column': [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22], 
            'Concentration': [0,0,0,0,10,10,30,30,100,100,300,300,1000,1000,3000,3000,10000,10000,30000,30000]}
    
    drug = {'Row':    [3,4,5,6,7,8,9,10,11,12,13,14], 
            'Compound': ['DRUG1','DRUG1','DRUG2','DRUG2','DRUG3','DRUG3','DRUG4','DRUG4','DRUG5','DRUG5','DRUG6','DRUG6']}
    
    concmap = pd.DataFrame(conc, columns=['Column','Concentration'])
    drugmap = pd.DataFrame(drug, columns=['Row','Compound'])

    df = pd.read_csv(path,skiprows=9,sep="\t")
    df = df.rename(columns=lambda x: x.replace('valid objects - ', ''))

    df.drop(columns=['Compound','Concentration'], inplace=True)
    df = df.merge(concmap,on='Column')
    df = df.merge(drugmap,on='Row')
    df.loc[(df['Column'] >= 3) & (df['Column'] <= 6),'Compound'] = 'DMSO'
    df.drop(columns=['Unnamed: 1098'], inplace=True, ignore_errors=True)
    return df

def AggregateProfiles(scData, file_type='harmony'):
    if file_type == 'harmony':
        # aggregate profiles per well on all features
        scData.drop(list(scData.filter(regex='.*(Image).*|.*(Position).*|.*(Centroid).*|.*(Distance).*')),
                    axis=1, inplace=True)  # drop positional features to avoid plot issues
        # extrqct list of calculated features
        scData = scData.rename(columns={"Object No": 'Metadata_ObjectNumber'}, errors='ignore')
        metaFeatureList = ['Row','Column','Metadata_Object_Count','Concentration','Compound']#df.columns[11:].to_list()
        featureList = scData.columns[11:].to_list()  # extract metadata columns
        # aggregate profiles per well on all features
        normalized = pcm.aggregate(scData, strata=["Row", "Column"], features=featureList, compute_object_count=True)
        metadataSINGE = scData.groupby(['Row','Column'],as_index=False).first()
        metadataSINGE = metadataSINGE[['Compound','Concentration']]
        normalized = pd.concat((metadataSINGE,normalized), axis=1)
        normalized = pcm.normalize(normalized, samples="Compound=='DMSO'", features=featureList,
                                    meta_features=metaFeatureList, method="mad_robustize")  # normalize using RobustMAD
        # add conditions to aggregated table
        normalized.set_index(['Compound', 'Concentration'], inplace=True)
        normalized.drop(['Row', 'Column'], axis=1, inplace=True)
        normalized = normalized.sort_values(by=['Compound', 'Concentration'])
        normalized = normalized.loc[:,~normalized.columns.duplicated()]
        normalized = normalized.drop(['Cell Type','Cell Count'], axis=1)
    elif file_type == 'cellprofiler':
        metaFeatureList = ['PlateID','Metadata_Well','Metadata_Concentration (Image)','Compound']
        featureList = list(set(scData.columns.to_list()) - set(metaFeatureList))
        normalized = pcm.normalize(scData, samples="Compound=='DMSO'", features=featureList,
                                    meta_features=metaFeatureList, method="mad_robustize", mad_robustize_epsilon=0)  # normalize using RobustMAD
    else:
        raise ValueError('file_type must be either "harmony" or "cellprofiler"')
    return normalized

def drop_correlated_columns(df, threshold, numeric):
    # drop columns with correlation greater than threshold, numeric = True if only numeric columns are to be considered
    nbFeaturesStart = df.shape[1]
    # Create correlation matrix
    corr_matrix = df.corr(numeric_only=numeric).abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Find index of feature columns with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    # Drop features 
    filtered = df.drop(df[to_drop], axis=1)
    nbFeaturesEnd = filtered.shape[1]
    diff = nbFeaturesStart-nbFeaturesEnd
    print("Correlation filter : "+str(diff) 
          +" features dropped ("+str(round(100*diff/nbFeaturesStart))+"% of total)")
    return filtered

def drop_high_variance(df, threshold):
    # drop features with variance greater than threshold
    nbFeaturesStart = df.shape[1]
    var = pd.DataFrame(df.var(numeric_only=True))
    if 'PlateName' in df.columns:
        df.set_index(['Compound','Concentration','PlateName'],inplace=True,drop=True,append=True)
    else:
        df.set_index(['Compound','Concentration'],inplace=True,drop=True,append=True)
    i = len(df.index)-1
    metadata = pd.concat([df.iloc[:,0]],axis=1)
    data = df.iloc[:,1:i]
    var = pd.DataFrame(data.std(numeric_only=True))
    filtFeatures = var.loc[var[0] < threshold].index.tolist()
    data = data.filter(items=filtFeatures,axis=1)
    profFiltered = pd.concat([metadata,data],axis=1)
    
    profFiltered.reset_index(inplace=True)
    profFiltered.drop('level_0',axis=1, inplace=True, errors='ignore')
    nbFeaturesEnd = profFiltered.shape[1]
    diff = nbFeaturesStart - nbFeaturesEnd
    print("High variance filter : "+str(diff) +" features dropped ("+str(round(100*diff/nbFeaturesStart, 3))+"%)")
    return profFiltered

def drop_low_variance(df, threshold=0.1, file_type='harmony'):
    if file_type == 'harmony':
    # drop features with variance less than threshold
        nbFeaturesStart = df.shape[1]
        var = pd.DataFrame(df.var(numeric_only=True))
        if 'PlateName' in df.columns:
            df.set_index(['Compound','Concentration','PlateName'],inplace=True,drop=True,append=True)
        else:
            df.set_index(['Compound','Concentration'],inplace=True,drop=True,append=True)
        i = len(df.index)-1
        metadata = pd.concat([df.iloc[:,0]],axis=1)
        data = df.iloc[:,1:i]
        var = pd.DataFrame(data.var(numeric_only=True))
        filtFeatures = var.loc[var[0] > threshold].index.tolist()
        data = data.filter(items=filtFeatures,axis=1)
        profFiltered = pd.concat([metadata,data],axis=1)

        profFiltered.reset_index(inplace=True)
        profFiltered.drop('level_0',axis=1, inplace=True, errors='ignore')
        nbFeaturesEnd = profFiltered.shape[1]
        diff = nbFeaturesStart - nbFeaturesEnd
        print("Low variance filter : "+str(diff) +" features dropped ("+str(round(100*diff/nbFeaturesStart, 3))+"%)")
    elif file_type == 'cellprofiler':
        nbFeaturesStart = df.shape[1]
        df.set_index(['Metadata_Well','Compound','Metadata_Concentration (Image)','PlateID'],inplace=True,drop=True,append=True)
        var = pd.DataFrame(df.var(numeric_only=True))
        filtFeatures = var.loc[var[0] > threshold].index.tolist()
        df = df.filter(items=filtFeatures,axis=1)
        df.reset_index(inplace=True)
        nbFeaturesEnd = df.shape[1]
        diff = nbFeaturesStart - nbFeaturesEnd
        print("Low variance filter : "+str(diff) +" features dropped ("+str(round(100*diff/nbFeaturesStart, 3))+"%)")
        profFiltered = df
        profFiltered.drop('level_0',axis=1, inplace=True, errors='ignore')
    else:
        raise ValueError('file_type must be either "harmony" or "cellprofiler"')

    return profFiltered


def StandardProfileFormatting(prof):
    #formatting per well profiles from Harmony data
    prof.drop(columns='Unnamed: 0', inplace=True)
    std = pd.DataFrame(prof.std(numeric_only=True))
    prof = prof.drop(columns=['PlateName'])
    prof.set_index(['Compound','Concentration'],inplace=True,drop=True,append=True)
    prof.head()
    #count = prof.pop('Count')
    i = len(prof.index)-1
    metadata = pd.concat([prof.iloc[:,0]],axis=1)
    data = prof.iloc[:,1:i]
    std = pd.DataFrame(data.std(numeric_only=True))
    filtFeatures = std.loc[std[0] < 1000].index.tolist()
    data = data.filter(items=filtFeatures,axis=1)

    profFiltered = pd.concat([metadata,data],axis=1)
    profFiltered.reset_index(inplace=True)
    profFiltered.set_index(['Compound','Concentration'])
    profFiltered.drop('level_0',axis=1)
    return profFiltered

def ComputeSampleInduction(df):
    # Create a new column 'Induction' and initialize it with 0
    df['Induction'] = 0
    # Iterate through the rows of the dataframe to compute the induction percentage
    for i, row in df.iterrows():
        count = 0
        # Iterate through the values in the row
        for value in row:
            if value > 3 or value < -3:
                count += 1
        indRatio = 100 * count / len(df.columns)
        # Assign the count to the corresponding row in the 'count' column
        df.at[i, 'Induction'] = indRatio
    return df



class FeatureColorMapping():
    # return three series containing color coding corresponding to the features regions, channels, and classes

    # Create lists to store the column regions, channels, and classes

    def getColors(self, df, file_type):
        if file_type == 'harmony':

            region_list = ['Cell ', 'Nucleus ', 'Cytoplasm ', 'Ring ', 'Other']
            channel_list = ['33342 ','488 ','568 ','555 ','Mito ', 'Other']
            class_list = ['Shape','Texture','Intensity','SER','Symmetry', 'Other']
            
        elif file_type == 'cellprofiler':

            region_list = ['Cells', 'Nuclei', 'Cytoplasm','Other']
            channel_list = ['MITO', 'DNA', 'ER', 'RNA', 'AGP', 'Other']
            class_list = ['AreaShape', 'Granularity', 'Intensity', 'Location', 'Neighbors','RadialDistribution', 'Texture', 'Correlation','Other']
        else:
            raise ValueError('file_type must be either "harmony" or "cellprofiler"')

        regions, channels, classes = [], [], []
        # Iterate through the column names
        for col in df.columns:
            regionFound, channelFound, classFound = False, False, False
            # Iterate through the region list
            for region_name in region_list:
                if region_name in col and not regionFound:
                    regions.append(region_name)
                    regionFound = True
                    break
            # Iterate through the channel list
            for channel_name in channel_list:
                if channel_name in col and not channelFound:
                    channels.append(channel_name)
                    channelFound = True
                    break
            # Iterate through the class list
            for class_name in class_list:
                if class_name in col and not classFound:
                    classes.append(class_name)
                    classFound = True
                    break
            if ("Profile" in col or "Axial" in col) and not classFound:
                classes.append("Shape")
                classFound = True

            if not regionFound:
                regions.append('Other')
                regionFound = True
            if not channelFound:
                channels.append('Other')
                channelFound = True
            if not classFound:
                classes.append('Other')
                classFound = True

        region_map = {region:color for region, color in zip(region_list, sns.color_palette("Set2", len(region_list)))}
        channel_map = {channel:color for channel, color in zip(channel_list, sns.color_palette("Set1", len(channel_list)))}
        class_map = {class_name:color for class_name, color in zip(class_list, sns.color_palette("Set3", len(class_list)))}

        regionColors = [region_map[region] for region in regions]
        channelColors = [channel_map[channel] for channel in channels]
        classColors = [class_map[i] for i in classes]

        return regionColors, channelColors, classColors

    def getFeatureTypes(self, file_type):
        if file_type == 'harmony':
            class_list = ['Shape','Texture','Intensity','SER','Symmetry', 'Other']
            channel_list = ['33342 ','488 ','568 ','555 ','Mito ', 'Other']
            region_list = ['Cell ', 'Nucleus ', 'Cytoplasm ', 'Ring ', 'Other']
        elif file_type == 'cellprofiler':
            class_list = ['AreaShape', 'Granularity', 'Intensity', 'Location', 'Neighbors','RadialDistribution', 'Texture', 'Correlation','Other']
            channel_list = ['MITO', 'DNA', 'ER', 'RNA', 'AGP', 'Other']
            region_list = ['Cells', 'Nuclei ', 'Cytoplasm','Other']
        else:
            raise ValueError('file_type must be either "harmony" or "cellprofiler"')
        print(file_type + ' features selected')
        return region_list, channel_list, class_list

    def getFeatureMaps(self, region_list, channel_list, class_list):
        region_map = {region:color for region, color in zip(region_list, sns.color_palette("Set2", len(region_list)))}
        channel_map = {channel:color for channel, color in zip(channel_list, sns.color_palette("Set1", len(channel_list)))}
        class_map = {class_name:color for class_name, color in zip(class_list, sns.color_palette("Set3", len(class_list)))}
        return region_map, channel_map, class_map
    
def consistent_file_format(prof):
        # return a consistent file format for the input dataframe
        if 'Metadata_Concentration (Image)' in prof.columns:
            prof.rename(columns={'Metadata_Concentration (Image)':'Concentration', 'PlateID':'PlateName'}, inplace=True)
            prof.drop('Unnamed: 0', axis=1, inplace=True)
            prof['Count'] = prof['Count']*5
            # add a "-" character at position -4 to the plate name to match the LOAEL file
            prof['PlateName'] = prof['PlateName'].str[:-4] + '-' + prof['PlateName'].str[-4:]
            file_type = 'cellprofiler'
        else:
            file_type = 'harmony'
        return prof

def cp_feature_classifier(df, file_type='harmony', with_colors=False):
    # Returns a dataframe containing information about the feature type

    # Input: dataframe, file type (harmony or cellprofiler)
    # Output: dataframe with columns 'Region', 'Channel', 'Class'

    if file_type == 'harmony' or file_type == 'jump':

        region_list = ['Cell', 'Nucleus', 'Cytoplasm', 'Ring', 'Other','Nuclei']
        channel_list = ['33342 ','488 ','568 ','555 ','647 ','Mito','RNA','ER','AGP','DNA']
        class_list = ['Texture','Intensity','Symmetry','Axial', 'Compactness','Radial','Profile','SER','Correlation','Granularity','AreaShape','Neighbors']
    
        regions, channels, classes = [], [], []
        # Iterate through the column names
        for col in df.columns:
            regionFound, channelFound, classFound = False, False, False
            # Iterate through the region list
            for region_name in region_list:
                if region_name in col and not regionFound:
                    regions.append(region_name)
                    regionFound = True
                    break
            # Iterate through the channel list
            for channel_name in channel_list:
                if channel_name in col and not channelFound:
                    channels.append(channel_name)
                    channelFound = True
                    break
            # Iterate through the class list
            for class_name in class_list:
                if class_name in col and not classFound:
                    classes.append(class_name)
                    classFound = True
                    break
            if not regionFound:
                regions.append('Other')
            if not channelFound:
                channels.append('Other')
            if not classFound:
                classes.append('Other')
        #create a dataframe with feature type, region, and channel, and a last column with the three concatenated with underscores
        featureTypeTable = pd.DataFrame({'Feature':df.columns, 'Region':regions, 'Channel':channels, 'Class':classes}).sort_values(by=['Channel','Class','Region'])
        # rename channels to match cellprofiler
        featureTypeTable['Channel'] = featureTypeTable['Channel'].replace({'33342 ':'DNA', '488 ':'AGP', '568 ':'RNA', '555 ':'ER', '647 ':'Mito', 'MITO':'Mito', })
        featureTypeTable['Class'] = featureTypeTable['Class'].replace({'SER':'Texture'})
        featureTypeTable['Region'] = featureTypeTable['Region'].replace({'Nucleus':'Nuclei', 'Cell':'Cells'})
        featureTypeTable['FeatureCategory'] = featureTypeTable['Region'] + '_' + featureTypeTable['Class'] + '_' + featureTypeTable['Channel']

    elif file_type == 'cellprofiler':
    
        featureList = df.columns.to_list()

        featureTypeList = [x.split('_')[0] for x in featureList]
        featureRegionList = [x.split(' ')[1][1:-1] if ' ' in x else 'other' for x in featureList]
        featureChannelList = [x.split('_')[2] if len(x.split('_'))==3 else x.split('_')[3] if len(x.split('_'))>=3 else "Global" for x in featureList] 

        for x in featureList:
            if (len(x.split('_')) == 3) | (len(x.split('_')) >= 5 | ( (len(x.split('_')) == 4) & (x.split('_')[0] == 'RadialDistribution'))):
                featureChannelList[featureList.index(x)] = x.split('_')[2]
            elif len(x.split('_'))==4:
                featureChannelList[featureList.index(x)] = x.split('_')[3]
            elif x.split('_')[0] == ['Texture'] :
                featureChannelList[featureList.index(x)] = x.split('_')[2]
            else:
                featureChannelList[featureList.index(x)] = "Global"

        featureChannelList = [x.split(' ')[0] if (len(x.split(' '))>=2) else x for x in featureChannelList]
        featureChannelList = [x if x in ['AGP','DNA','ER','MITO','RNA'] else 'Global' for x in featureChannelList]

        #create a dataframe with feature type, region, and channel
        featureTypeTable = pd.DataFrame({'Feature':featureList,'Region':featureRegionList, 'Class':featureTypeList, 'Channel':featureChannelList})

        # merge columns Region, Channel and Class into a new column FeatureType2, separated by _
        featureTypeTable["FeatureCategory"] = featureTypeTable["Region"].str.cat([featureTypeTable["Class"], featureTypeTable["Channel"]], sep="_")

        # handle the case where one of the columns is not found ('Other'  )
        featureTypeTable.loc[featureTypeTable["FeatureCategory"].str.contains("Other"), "FeatureCategory"] = featureTypeTable.loc[featureTypeTable["FeatureCategory"].str.contains("Other"), ["Region","Class"]].apply(lambda x: "_".join(x.dropna()), axis=1)

    else:
        raise ValueError('file_type must be either "harmony" or "cellprofiler"')
    
    if with_colors:
        # create the color maps for regions, classes, and channels
        region_list = list(set(featureTypeTable['Region']))
        region_map = {region:color for region, color in zip(region_list, sns.color_palette("Set2", len(region_list)))}

        channel_list = list(set(featureTypeTable['Channel']))
        channel_map = {channel:color for channel, color in zip(channel_list, sns.color_palette("Set1", len(channel_list)))}

        class_list = list(set(featureTypeTable['Class']))
        class_map = {class_name:color for class_name, color in zip(class_list, sns.color_palette("Set3", len(class_list)))}

        # add the Region_Color, Class_Color, and Channel_Color columns to the featureTypeTable
        featureTypeTable['Region_Color'] = featureTypeTable['Region'].map(region_map).apply(list)
        featureTypeTable['Class_Color'] = featureTypeTable['Class'].map(class_map).apply(list)
        featureTypeTable['Channel_Color'] = featureTypeTable['Channel'].map(channel_map).apply(list)

    return featureTypeTable


def get_file_type(table):
    # check if the input dataframe is from Harmony or CellProfiler
    if "Metadata_Concentration (Image)" in table.columns:
        file_type = "cp"
        table.rename(columns={'Metadata_Compound':'Compound', 'Metadata_Concentration (Image)':'Concentration','Metadata_Well':'Well'}, inplace=True)
        table.drop(columns=['PlateID','Well'], inplace=True, errors='ignore')
    else:
        table.drop(columns=['PlateName'], inplace=True, errors='ignore')
        file_type = "harmony"
    return file_type

def get_cell_type(file_name):
    match file_name:
        case "tubesMonoByWellProfiler":
            return "TubesMono"
        case "tubesPolyByWellProfiler":
            return "TubesPoly"
        case "blastsAllByWellProfiler":
            return "Blasts"
        case "blastsProfiles":
            return "Blasts"
        case "tubesProfiles":
            return "Tubes"
        case _:
            raise ValueError(f"Unknown cell type: {file_name!r}")
        
def AddDrugNames(df, col='Concentration', row='Compound', well='Well'):
    # add drug names to the dataframe
    conc = {'Column': [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22], 
            'Concentration': [0,0,0,0,10,10,30,30,100,100,300,300,1000,1000,3000,3000,10000,10000,30000,30000]}
    
    drug = {'Row':    [3,4,5,6,7,8,9,10,11,12,13,14], 
            'Compound': ['DRUG1','DRUG1','DRUG2','DRUG2','DRUG3','DRUG3','DRUG4','DRUG4','DRUG5','DRUG5','DRUG6','DRUG6']}
    
    concmap = pd.DataFrame(conc, columns=['Column','Concentration'])
    drugmap = pd.DataFrame(drug, columns=['Row','Compound'])

    df = df.merge(concmap,on='Column')
    df = df.merge(drugmap,on='Row')
    df.drop(columns=['Compound','Concentration'], inplace=True)
    df = df.merge(concmap,on='Column')
    df = df.merge(drugmap,on='Row')
    df.loc[(df['Column'] >= 3) & (df['Column'] <= 6),'Compound'] = 'DMSO'
    

    
    return df