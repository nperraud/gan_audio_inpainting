# Week 3: 18 october 2019


## This week tasks

Unsderstand overfitting
	* Prepare the validation set for the piano data

A) Time domain with maestro
    * https://magenta.tensorflow.org/datasets/maestro
  	* Prepare the dataset
  	  1. Cut in training/testing
  	  2. Prepare the data for the batches (2 secs, 1 sec, 2 secs) 
  	* Find a way to feed it into the network...

B) Improve the global architecture
	* Get some data that should be bigger than the piano but not too big (possible source: youtube)
	* Understand the file `code/gantools/gantools/model.py` (class InpaintingGAN)
	* Make a new class similar to InpaintingGAN
		* with the context block
	* Steps:
		1. Understand the model code
		2. Draw current architecture
		3. Draw modification
		4. Check with Nathanaël if ok
		5. Code	



## Last week tasks
1. Try to execute/understand the code
2. Try to run stuff on CSCS
3. Search for another dataset than piano (you can use maybe the free music archive)
4. Read the WaveGAN and TifGAN papers, maybe also Wavenet
5. Change a bit the architecture of the demo `GAN-audio-inpainting.ipynb` and try to overfit it.
	* To check overfitting, you need to create a test set. -> similar to the training