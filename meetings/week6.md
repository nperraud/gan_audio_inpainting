# Week 6: 21 November 2019

## Pirmin: this week tasks

- Advice:
```
def gen_data(files):
'''Return one data point that goes into the network'''
  for file in files:
    dat = load(file)
    datp = preprocess(dat) # augmentation and slicing
    for d in datp:
      yield d
def grouper(gen, batchsize):
  ...
def queued_generator(...)
  ...

gen_slow = gen_data(files)
gen_bs = grouper(gen_slow, batch_size)
gen_fast_bs = queued_geneartor(gen_bs)
```
- Have a trained network
  * assess the performance...

## Amr: this week tasks
  - Run the code with the Solo Dataset
  - Try to implement the double discriminator model.
    * Go step by step


## Amr: last week tasks

Part A: Solo Dataset
    - Check the `Dataset` class is in `gantools/data/core.py`
    - Understand what is the sampling frequency, downsampling data
    - Make the dataset
        * Retrain the model with the new data from the `solo` dataset

Part B: Improve the global architecture

  * Steps:
    1. Draw current architecture
    2. Draw modification
    3. Check with Nathanaël if ok
    4. Understand the model code
    5. Code the new architecture in a new file...

  * Understand the file `code/gantools/gantools/model.py` (class InpaintingGAN)
  * Make a new class similar to InpaintingGAN
    * with the context block

## Pirmin: last week tasks

- Build a generator for the large dataset... (all the pipeline including some function to download the files)

- Time domain with maestro
  	* Prepare the dataset
  	  1. Cut in training/testing
  	  2. Prepare the data for the batches (2 secs, 1 sec, 2 secs) 
  	* Find a way to feed it into the network...
    
