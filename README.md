# NOTES ANNA

Note: besides inference scripts, most scripts will have to be moved from their respective folder into the main folder to be used, or the relevant files (like models.py for UNet) will have to be used. For specific models, their names may have changed.
what scripts to use:

- data
    - the main datasets currently in use can be found at https://huggingface.co/datasets/AnnaStuckert/facemap_dolensek2020/tree/main 

- GradCAM tutorial
    - this includes the work conducted on making GradCAM work for Unet segmentation model. the jupyter notebook is the file mainly in use. the model files are put there to load the corresponding unet structure.

- make_dataset_files
    - this folder contains script for making dataset files. these take the raw files from DLC (images and .csv files with labels) to create .pt dataset formats
    - combine_facemap_dolensek - this is used to take the .pt files for dolensek2020 and facemap datasets and 1) combine them to one big dataset and 2) make soft labels. contains code both for the normal soft labels between 0 and 1, and now also soft labels in terms of having values per KP, so a segmentation mask of 1 for KP 1, 2 for KP 2, etc.
    - make_dataset_w_changes this one works for generating .pt files when going into a single folder 9so not containing multiple subfolders with .csv labels in each subfolder. There should be a script which can do this - that is the KP_checker_and_make_dataset_cleaned!!!.
    - !!! KP_checker_and_make_dataset_cleaned and KP_checker_and_make_dataset_cleaned_dolensek I believe was used to check that KP placement was correct (for schroeder data), then applied to the dolensek data. this can go into multiple subfolders. both includes raw DLC to .pt KP format, and to softlabels.
    - make_softlabels turns the KP .pt files into soft gaussian labels for segmentation Unet
    - make_dataset_RGB and make_dataset_RGB2 - I believe these were used to create the .pt dataset format but with RGB instead of single channel images. might be deleted.


- Inference scripts folder contains scripts for video inference
    - Video_inference_segmentation_model_dolensek.ipynb is the newest version. uses the Unet trained model to make predictions. works with the combined facemap and dolensek datasets with soft labels. 
    - Video_inference_segmentation_model.ipynb should do the same, but runs into a problem. have not worked further on accessing the problem. can most likely be deleted. - moved to scrapped scripts.
    - Video_inference_windows.ipynb contains inference script for ViT inference on a windows computer

- unet segmentation scripts
    - train_seg is the original unet segmentation script from Raghav
    - train_seg_2 - not too sure what the differnce is for this script, might be deleted
    - train_seg_more_aug_CURRENTLY_IN_USE is the currently used segmentation unet training script, includes a bunch of data augmentations. currently placed in the main folder

- Regression transformer - original facemap training scripts using ViT
    - this folder contains scripts used in the original ViT trainig scripts from Raghav for KP regression.
    - train_filter_nan_dataaug_USE_THIS - this is the most recent training script, it filters items in the dataset that has NaNs. also includes some data augmentations
    - train_filter_nan - I think this does the above script, but with limited data augmentation
    - train.py - this was the original training script from Raghav, it did not exclude NaNs but coverted to 0, and should exclude those from affecting the loss.
    - train_vit_moreaug - i think this covers more data augmentations
    - OBs for some of the original ViT scripts form raghav, if not including plt.close() at end of especially the logs visualization, the model crashes at approx 263 epochs. 
    - OBS this code should ensure NaNs (converted to 0s) are not used in computing the loss, but subsequent scripts I just filtered out NaN instead for clarity.         loss = loss_fun(
            torch.log(scores[labels != 0]), torch.log(F.softplus(labels[labels != 0]))


- Segmentation transformer - ViT and SWIN work
    - this folder contains scripts from the ViT for segmentation (mainly the SWIN transformer) - this is largely regarded as a failed attempt but kept for reference

# README #

### What is this repository for? ###

* Reproduce results in the paper. 
* v1.0

### How do I get set up? ###

* Basic Pytorch dependency
* Tested on Pytorch 2.0, Python 3.8


### Usage guidelines ###

* Kindly cite our publication if you use any part of the code

### Who do I talk to? ###

* raghav@di.ku.dk

