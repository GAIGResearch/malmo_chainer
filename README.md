 ## --- This is a draft document ---

## Introduction
Marlo (short for Multi-Agent Reinforcement Learning in Malmo) is an artificial intelligence competition primarily aimed towards the goal of implementing reinforcement learning agents with a great degree of generality, capable of solving problems in pseudo-random, procedurally changing multi-agent environments within the world of the mediatic phenomenon game Minecraft.

Marlo is based off of the [Malmo](https://github.com/Microsoft/malmo) framework, which is a platform for Artificial Intelligence experimentation and research built on top of Minecraft. The Malmo platform provides a high-level API which enables access to actions, observations (i.e. location, surroundings, video frames, game statistics) and other general data that Minecraft provides. Marlo, on the other hand, is a wrapper for Malmo that provides a more standardized RL-friendly environment for scientific study.

The framework is written as an extension to [OpenAI's Gym](https://github.com/openai/gym) framework, which is a toolkit for developing and comparing reinforcement learning algorithms, thus providing an industry-standard and familiar platform for scientists, developers and popular RL frameworks.

Due to the framework's nature as an wrapper for Malmo, a few steps must be taken in order to be able to boot up and run Marlo.

## Getting started
### Marlo installation with Malmo repack
1. Install a suitable version of Malmo
    * This can easily be done by following the step-by-step guide that Malmo provides for [Windows](https://github.com/Microsoft/malmo/blob/master/doc/install_windows.md), [Linux](https://github.com/Microsoft/malmo/blob/master/doc/install_linux.md), [MacOSX](https://github.com/Microsoft/malmo/blob/master/doc/install_macosx.md).
    * *Please make sure to download a pre-compiled version of Malmo as posted on the release page as doing this is ***not*** the same as downloading the GitHub repository ZIP file. If you choose to download the repository, you will have to ***build the package yourself*** which is a lengthier process. If you get errors along the lines of "ImportError: No module named MalmoPython" it will probably be because you have made this mistake.*
    * Please ensure that all the redistributables that Malmo requires are installed correctly, and that the following entries appear as environment variables:
        * MALMO_XSD_PATH = Malmo_dir\Schemas folder
        * JAVA_HOME = jdk installation folder
2. Clone Marlo repository from [this](#) GitHub repository's Master branch.
3. Navigate to the folder where you have downloaded the repository and run the setup.py file: python setup.py install
    * This should install all the required packages for Marlo to function.
    * In some special circumstances, some packages might fail to install, prompting you to install them yourself. When this happens, the error message usually contains the name and link to the missing packages which you should install using PyPi.
    * Note: if an import error describing a missing "scoretable" class appears, please downgrade Gym to version 0.7.4 like such: *pip install -U gym==0.7.4*
    
## Using our recommended framework, ChainerRL
In their own words, Chainer "is a Python-based deep learning framework aiming at flexibility". It has a very powerful high-level API aimed at training deep learning networks and as such is very useful in a RL context. ChainerRL is a deep reinforcement learning library that implements various state-of-the-art deep reinforcement algorithms in Python using Chainer.

ChainerRL is the Marlo's officially endorsed framework and as such examples for its usage will be posted: however, ChainerRL is by no means mandatory for the competition!

The framework presents a wide range of algorithms and deep learning tools which facilitate a quick start-up and as such is ideal for drafts. ChainerRL communicates seamlessly with OpenAI's Gym framework, thus relieving a lot of structural stress off of you - the competitor - and allowing you to focus strictly on your agent's behaviour.

### Installing and running ChainerRL
Please refer to ChainerRL's [official GitHub](https://github.com/chainer/chainerrl) for installation instructions and further documentation. Alternatively, you can simply use PyPi to download and install ChainerRL as a package via the following command: *pip install chainerrl*. Following this, you can simply proceed to testing it out via following the steps laid out in ChainerRL's official [getting started guide](https://github.com/chainer/chainerrl/blob/master/examples/quickstart/quickstart.ipynb)
      
