# Week 4: 23 october 2019

## Amr: this week tasks

- Check the `Dataset` class is in `gantools/data/core.py`

- Make the dataset
    * Retrain the model with the new data from the `solo` dataset

- Understand what is the sampling frequency, downsampling data

- Improve the global architecture
  * Understand the file `code/gantools/gantools/model.py` (class InpaintingGAN)
  * Make a new class similar to InpaintingGAN
    * with the context block
  * Steps:
    1. Understand the model code
    2. Draw current architecture
    3. Draw modification
    4. Check with Nathanaël if ok
    5. Code   


## Pirmin: this week tasks

- Build a generator for the large dataset... (all the pipeline including some function to download the files)

- Time domain with maestro
  	* Prepare the dataset
  	  1. Cut in training/testing
  	  2. Prepare the data for the batches (2 secs, 1 sec, 2 secs) 
  	* Find a way to feed it into the network...
    
