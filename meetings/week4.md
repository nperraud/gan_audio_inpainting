# Week 4: 23 october 2019


## This week tasks

- Unsderstand what is a generator in python
- One way to build the generator
```
# initialize the queue with a length of 2 (or 1...)
for _ in range(n_files):
    curr_data = queue.get()
    for ...
        yield batch

```
  https://stackoverflow.com/questions/43078980/python-multiprocessing-with-generator

- Time domain with maestro
  	* Prepare the dataset
  	  1. Cut in training/testing
  	  2. Prepare the data for the batches (2 secs, 1 sec, 2 secs) 
  	* Find a way to feed it into the network...
    

## Last week tasks

Unsderstand overfitting
	* Prepare the validation set for the piano data

A) Time domain with maestro
    * https://magenta.tensorflow.org/datasets/maestro
  	* Prepare the dataset
  	  1. Cut in training/testing
  	  2. Prepare the data for the batches (2 secs, 1 sec, 2 secs) 
  	* Find a way to feed it into the network...