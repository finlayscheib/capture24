import numpy as np
import pandas as pd 
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyts.image import GramianAngularField


## read annotations

anno_label_dict = pd.read_csv('data/capture24/annotation-label-dictionary.csv',
                              index_col='annotation', dtype='string')

#Helper functionm
def extract_windows(data, winsize='30s'):
    X, Y = [], []
    for t, w in data.resample(winsize, origin='start'):

        # Check window has no NaNs and is of correct length
        # 10s @ 100Hz = 1000 ticks
        if w.isna().any().any() or len(w) != 3000:
            continue

        x = w[['x', 'y', 'z']].to_numpy()
        y = w['label'].mode(dropna=False).item()
    
        X.append(x)
        Y.append(y)

    X = np.stack(X)
    Y = np.stack(Y)

    return X, Y


N=3000 # number of data points in each window
fig_w=4; fig_h=4;# figure width and height


path_data=r"C:\Users\Finla\OneDrive - University of Edinburgh\Diss\capture24\data\capture24" # data directory
path_GAF=r"C:\Users\Finla\OneDrive - University of Edinburgh\Diss\capture24\data\GAF" # where GAF images will be saved

#make GAF folder to save files into
os.makedirs(path_GAF, exist_ok=True)

# working directory for data

directory_contents = [i for i in os.listdir(path_data) if i.endswith('.csv.gz')] # get the data file names
print(f"Found {len(directory_contents)} files to process: {directory_contents}") # comment out at end

#Initialise GADF method
gadf = GramianAngularField(method='difference')
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

    X, Y = extract_windows(data) # extract data


    img_cnt=1
    img_label_cnt=0
    for i in X:
        acc_x=i[:,0] # get x from the array
        acc_y=i[:,1] # get y from the array
        acc_z=i[:,2] # get z from the array
        acc_VM=np.sqrt((acc_x**2) + (acc_y**2) + (acc_z**2)) #compute vecor magnitude
        

        figure_name_GAF_X='P'+f"{p_cnt:03}"+'_'+str(img_cnt)+'_GAF_X'+'_'+Y[img_label_cnt]+'.jpg'
        figure_name_GAF_Y='P'+f"{p_cnt:03}"+'_'+str(img_cnt)+'_GAF_Y'+'_'+Y[img_label_cnt]+'.jpg'
        figure_name_GAF_Z='P'+f"{p_cnt:03}"+'_'+str(img_cnt)+'_GAF_Z'+'_'+Y[img_label_cnt]+'.jpg'
        figure_name_GAF_VM='P'+f"{p_cnt:03}"+'_'+str(img_cnt)+'_GAF_VM'+'_'+Y[img_label_cnt]+'.jpg'

        save_path_X = os.path.join(path_GAF, figure_name_GAF_X)
        save_path_Y = os.path.join(path_GAF, figure_name_GAF_Y)
        save_path_Z = os.path.join(path_GAF, figure_name_GAF_Z)
        save_path_VM = os.path.join(path_GAF, figure_name_GAF_VM)
            
        #plot X
        fig, axs = plt.subplots(1, 1, figsize=(fig_w,fig_h))
        plt.axis('off')
        x = np.array([acc_x])

        X_gadf = gadf.fit_transform(x)

        axs.imshow(X_gadf[0],
                   cmap='rainbow',
                  origin='lower')
        axs.axis('off')
        plt.savefig(save_path_X, bbox_inches='tight',pad_inches = 0, dpi=100);
        print(j,img_cnt,figure_name_GAF_X) 
        fig.clf()
        plt.close(fig) 


        #plot Y
        fig, axs = plt.subplots(1, 1, figsize=(fig_w,fig_h))
        plt.axis('off')
        y = np.array([acc_y])
    
        Y_gadf = gadf.fit_transform(y)

        axs.imshow(Y_gadf[0],
                   cmap='rainbow',
                  origin='lower')
        axs.axis('off')
        plt.savefig(save_path_Y, bbox_inches='tight',pad_inches = 0, dpi=100);
        print(j,img_cnt,figure_name_GAF_Y) 
        fig.clf()
        plt.close(fig) 

        
        #plot Z
        fig, axs = plt.subplots(1, 1, figsize=(fig_w,fig_h))
        plt.axis('off')
        z = np.array([acc_z])
        
        z_gadf = gadf.fit_transform(z)

        axs.imshow(z_gadf[0],
                   cmap='rainbow',
                  origin='lower')
        axs.axis('off')
        plt.savefig(save_path_Z, bbox_inches='tight',pad_inches = 0, dpi=100);
        print(j,img_cnt,figure_name_GAF_Z) 
        fig.clf()
        plt.close(fig)
        
             
                
        #plot VM
        fig, axs = plt.subplots(1, 1, figsize=(fig_w,fig_h))
        plt.axis('off')
        v = np.array([acc_VM])
        
        v_gadf = gadf.fit_transform(v)

        axs.imshow(v_gadf[0],
                   cmap='rainbow',
                  origin='lower')
        axs.axis('off')
        plt.savefig(save_path_VM, bbox_inches='tight',pad_inches = 0, dpi=100);
        print(j,img_cnt,figure_name_GAF_VM) 
        fig.clf()
        plt.close(fig)     
                
        img_cnt+=1
        img_label_cnt+=1

    p_cnt+=1 
    
