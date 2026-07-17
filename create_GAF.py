import numpy as np
import pandas as pd 
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyts.image import GramianAngularField
from scipy.signal import resample_poly


## read annotations

anno_label_dict = pd.read_csv('data/capture24/annotation-label-dictionary.csv',
                              index_col='annotation', dtype='string')

#Helper functionm
def extract_windows(data, winsize='30s'):
    X, Y = [], []
    for t, w in data.resample(winsize, origin='start'):

        # Check window has no NaNs and is of correct length
        # 10s @ 100Hz = 1000 ticks
        if w.isna().any().any() or len(w) != 300: #30s at 10 Hz is 300 ticks
            continue

        x = w[['x', 'y', 'z']].to_numpy()
        y = w['label'].mode(dropna=False).item()
    
        X.append(x)
        Y.append(y)

    X = np.stack(X)
    Y = np.stack(Y)

    return X, Y


N=3000 # number of data points in each window
fig_w=2.24; fig_h=2.24;# figure width and height make 2.24 inches so that 224 by 224 images


path_data=r"/exports/eddie/scratch/s2190468/capture24/data/capture24/" # data directory
path_GAF=r"/exports/eddie/scratch/s2190468/capture24/data/GAF_ds/" # where GAF images will be saved

#make GAF folder to save files into
#os.makedirs(path_GAF, exist_ok=True)

# working directory for data

directory_contents = [i for i in os.listdir(path_data) if i.endswith('.csv.gz')] # get the data file names


#Initialise GADF method
gadf = GramianAngularField(image_size=224, method='difference')

p_cnt=1
for j in (directory_contents):
    print(f"Processing: {j}")
    
    file_to_read = os.path.join(path_data, j)
    # read data
    data = pd.read_csv(file_to_read, index_col='time', parse_dates=['time'],
                       dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'string'}) 
    #label data
    data['label'] = (anno_label_dict['label:Walmsley2020']
                 .reindex(data['annotation'])
                 .to_numpy())
    # Adding in downsampling step of the windows 100Hz to 10Hz
    tri_ax_raw = data[['x','y','z']].to_numpy()
    tri_ax_resample = resample_poly(tri_ax_raw, up=10, down=100, axis=0)
    
    labels_resampled = data['label'].iloc[::10].to_numpy()
    time_resampled = data.index[::10]
    
    min_len = min(len(tri_ax_resampled), len(labels_resampled))
        
    data = pd.DataFrame(
      tri_ax_resample[:min_len], 
      columns=['x', 'y', 'z'], 
      index=time_resampled[:min_len]
        )
    data['label'] = labels_resampled[:min_len]
    
    X, Y = extract_windows(data) # extract data
    
    participant_folder = f'P{p_cnt:03}' # Creates folder for each participant
    participant_path = os.path.join(path_GAF, participant_folder)
    os.makedirs(participant_path, exist_ok=True)
    
    
    img_cnt=1
    img_label_cnt=0
    for i in X:
      
        acc_x=i[:,0] # get x, y, z from the array
        acc_y=i[:,1]
        acc_z=i[:,2] 
        
        

        figure_name_GAF_X='P'+f"{p_cnt:03}"+'_'+f"{img_cnt:04}"+'_GAF_X'+'_'+Y[img_label_cnt]+'.jpg'
        figure_name_GAF_Y='P'+f"{p_cnt:03}"+'_'+f"{img_cnt:04}"+'_GAF_Y'+'_'+Y[img_label_cnt]+'.jpg'
        figure_name_GAF_Z='P'+f"{p_cnt:03}"+'_'+f"{img_cnt:04}"+'_GAF_Z'+'_'+Y[img_label_cnt]+'.jpg'
        

        save_path_X = os.path.join(participant_path, figure_name_GAF_X)
        save_path_Y = os.path.join(participant_path, figure_name_GAF_Y)
        save_path_Z = os.path.join(participant_path, figure_name_GAF_Z)
        

            
        #plot X
        x = np.array([acc_x])
        X_gadf = gadf.fit_transform(x)

        plt.imsave(save_path_X, X_gadf[0], cmap='rainbow', origin='lower')

        #plot Y
        y = np.array([acc_y])
        Y_gadf = gadf.fit_transform(y)

        plt.imsave(save_path_Y, Y_gadf[0], cmap='rainbow', origin='lower')

        
        #plot Z
        z = np.array([acc_z])
        Z_gadf = gadf.fit_transform(z)

        plt.imsave(save_path_Z, Z_gadf[0], cmap='rainbow', origin='lower')
        
                
        img_cnt+=1
        img_label_cnt+=1

    p_cnt+=1 
    
