# Audio inpainting with generative adversarial network

This is the repository for the DAS project `Audio inpainting with Generative Adversarial Networks`. Information about the project can be found in `tex/project_infos/`

Please keep this repository as clean as possible and do not commit data nor notebook with excecuted cells.

## How to use the code in this projects

1. Go the folder code 

	```
	cd code
	```

2. Initializae submodules

	```
	git submodule update --init --recursive
	```

3. Install package (make a virtual environnement first)

	```
	pip install -r requirements.txt
	```

	You may want to use the nogpu version of the packages (`requirements_nogpu.txt`) for you local computer.

4. Download datasets (just made it work for one)
	
	```
	python download_data.py
	python make_piano_dataset.py
	```

5. Launch jupyter
	
	```
	jupyter lab
	```



## Project general informations

* Meeting time: Thursday 2pm
* Students: Ebner Pirmin, Amr Eltelt
* Supervisor: Nathanaël Perraudin


## Previous work

#### Previous work on audio inptainting

* Deep learning based methods
  - A context encoder for audio inpainting: <https://arxiv.org/pdf/1810.12138.pdf>

* Non deeplearning methods
  - Audio declipping with social sparsity: <https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6853863>
  - LPC : <http://ant-s4.hsu-hh.de/dafx/papers/DAFX02_Kauppinen_Roth_signal_extrapolation.pdf>

* Engineering methods
  - Inpainting of long audio segments with similaritygraphs: <https://arxiv.org/pdf/1607.06667.pdf>
  - You can also check the demo: <https://lts2.epfl.ch/web-audio-inpainting/>

* More related work: Check the *related work* setion of 
  - <https://arxiv.org/pdf/1810.12138.pdf>
  - <https://arxiv.org/pdf/1607.06667.pdf>

#### Previous work on audio generation using GAN

Mostly you need to be aware of 

* WaveGAN <https://arxiv.org/pdf/1802.04208.pdf>, code at <https://github.com/chrisdonahue/wavegan>.
* TiFGAN <https://arxiv.org/pdf/1902.04072.pdf>, code <https://github.com/tifgan/stftGAN>, website <https://tifgan.github.io/>
* GANSythn (Maybe) <https://magenta.tensorflow.org/gansynth>

#### Important architectures for audio generation

* Wavenet: <https://deepmind.com/blog/article/wavenet-generative-model-raw-audio>
* Many papers...


## Data sources

- [Speech Commands Zero through Nine (SC09)](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/sc09.tar.gz)
- [Drum sound effects](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/drums.tar.gz)
- [Bach piano performances](http://deepyeti.ucsd.edu/cdonahue/wavegan/data/mancini_piano.tar.gz)
- [Nsynth dataset](https://magenta.tensorflow.org/datasets/nsynth)


## Code sources

* The main inspiration for the code is: <https://github.com/nperraud/CodeGAN>
* Notebook to start working: <https://github.com/nperraud/CodeGAN/blob/audio-inpainting/audio_experiment/GAN-audio-inpainting.ipynb>
* Main git used as a submodule for gan <https://github.com/nperraud/gantools/>

