### --------- load modules -------------------#
import sys
import os

Utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '/u/project/ngarud/Garud_lab/DANN/Utils/'))
sys.path.append(Utils_path)

from CNN_multiclass_data_mergeSims_A100 import *
import gc


model_name='CNN_bw_multiclass_sims_trained'

path_train="/u/project/ngarud/Garud_lab/DANN/DANNcolor/ProcessingData/ProcessedMayaSims/"
### LOAD DATA
# Memory-map the saved file
mmap_neutral = np.load(path_train+'Neu_sims.npy', mmap_mode='r')[:40000,:,:,0:1]
mmap_HS = np.load(path_train+'HS_sims.npy', mmap_mode='r')[:40000,:,:,0:1]
mmap_SS = np.load(path_train+'SS_sims.npy', mmap_mode='r')[:40000,:,:,0:1]

print(mmap_neutral.shape)

### --------- Parameters -------------------#
initial_loss_weights=[1,0]
val_split=0.1
batch_size=64 #64, 32

### --------- Build and train model -------------------#

print("BUILDING MODEL")
model = create_model(mmap_neutral)


print(model.summary())

#train model
print("TRAINING MODEL")
model,score=train_model(model,mmap_neutral,mmap_HS,mmap_SS,val_split=val_split,batch_size=batch_size,path=model_name) 

