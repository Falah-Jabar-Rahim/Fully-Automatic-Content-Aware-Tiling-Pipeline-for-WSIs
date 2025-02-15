import os
import pickle as pkl

import numpy as np
import pandas as pd

from options.test_options import TestOptions

opt = TestOptions().parse()  # get test options
# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 0
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1 

with open(os.path.join(opt.results_dir, opt.name,f"{opt.version}_test_latest", f'{opt.name}_ref_remove_imagemetrics.pkl'), 'rb') as f:
    all_calc = pkl.load(f)

with open(os.path.join(opt.results_dir, opt.name,f"{opt.version}_test_latest", f'{opt.name}_filtered_indx.pkl'),"rb") as f:
    ink_index = pkl.load(f)

with open(os.path.join(opt.results_dir, opt.name,f"{opt.version}_test_latest", f'{opt.name}_labels.pkl'), 'rb') as f:
    all_labels = pkl.load(f)

print(f"Length of all_calc : {len(all_calc)}")
print(f"Length of ink_index : {len(ink_index)}")
print(f"Length of all_labels : {len(all_labels)}")

#Table Stats
all_calc = np.array(all_calc)
ink_index = np.array(ink_index)
all_labels = np.array(all_labels)[:len(all_calc)]

#Full stats
print("All tissue patches")
print("Original")
print("SSIM: {} + {}".format(np.mean(all_calc[:,0,0]),np.std(all_calc[:,0,0])))
print("PSNR: {} + {}".format(np.mean(all_calc[:,0,1]),np.std(all_calc[:,0,1])))
print("VIF: {} + {}".format(np.mean(all_calc[:,0,2]),np.std(all_calc[:,0,2])))
print("Restored")
print("SSIM: {} + {}".format(np.mean(all_calc[:,1,0]),np.std(all_calc[:,1,0])))
print("PSNR: {} + {}".format(np.mean(all_calc[:,1,1]),np.std(all_calc[:,1,1])))
print("VIF: {} + {}".format(np.mean(all_calc[:,1,2]),np.std(all_calc[:,1,2])))
print("\n")

#Inked tissues
indices = np.where(all_labels==1)[0]
print("Inked tissue Patches")
print("Original")
print("SSIM: {} + {}".format(np.mean(all_calc[indices,0,0]),np.std(all_calc[indices,0,0])))
print("PSNR: {} + {}".format(np.mean(all_calc[indices,0,1]),np.std(all_calc[indices,0,1])))
print("VIF: {} + {}".format(np.mean(all_calc[indices,0,2]),np.std(all_calc[indices,0,2])))
print("Restored")
print("SSIM: {} + {}".format(np.mean(all_calc[indices,1,0]),np.std(all_calc[indices,1,0])))
print("PSNR: {} + {}".format(np.mean(all_calc[indices,1,1]),np.std(all_calc[indices,1,1])))
print("VIF: {} + {}".format(np.mean(all_calc[indices,1,2]),np.std(all_calc[indices,1,2])))
print("\n")
#Clean tissues
indices = np.where(all_labels==0)[0]
print("Clean tissue Patches")
print("Original")
print("SSIM: {} + {}".format(np.mean(all_calc[indices,0,0]),np.std(all_calc[indices,0,0])))
print("PSNR: {} + {}".format(np.mean(all_calc[indices,0,1]),np.std(all_calc[indices,0,1])))
print("VIF: {} + {}".format(np.mean(all_calc[indices,0,2]),np.std(all_calc[indices,0,2])))
print("Restored")
print("SSIM: {} + {}".format(np.mean(all_calc[indices,1,0]),np.std(all_calc[indices,1,0])))
print("PSNR: {} + {}".format(np.mean(all_calc[indices,1,1]),np.std(all_calc[indices,1,1])))
print("VIF: {} + {}".format(np.mean(all_calc[indices,1,2]),np.std(all_calc[indices,1,2])))
print("\n")
#Filtered tissues
ink_indices = ink_index
clean_indices = np.array(list(set(np.arange(len(all_calc)))-set(ink_indices)))
print("Filtered tissue Patches")
print("Original")
print("SSIM: {} + {}".format(np.mean(all_calc[:,0,0]),np.std(all_calc[:,0,0])))
print("PSNR: {} + {}".format(np.mean(all_calc[:,0,1]),np.std(all_calc[:,0,1])))
print("VIF: {} + {}".format(np.mean(all_calc[:,0,2]),np.std(all_calc[:,0,2])))
print("Restored")
form_list = np.concatenate((all_calc[ink_indices,1,:],all_calc[clean_indices,0,:]))
print("SSIM: {} + {}".format(np.mean(form_list[:,0]),np.std(form_list[:,0])))
print("PSNR: {} + {}".format(np.mean(form_list[:,1]),np.std(form_list[:,1])))
print("VIF: {} + {}".format(np.mean(form_list[:,2]),np.std(form_list[:,2])))

print("Done")