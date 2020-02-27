
# RL_Multy_Agent_Unity

### Running python code on existing build:

- unzip the build of required platform.

run main.py train -h to see options
example:

running test with example weights (windows):

	python main.py test --build ../build_windows/RaceGame_MARL.exe --num-agents 1 --agent ddpg --weights-path example_weights/ddpg_1_agent
	
	python main.py test --build ../build_windows/RaceGame_MARL.exe --num-agents 5 --agent mddpg --weights-path example_weights/mddpg_5_agents



basic run:

    python ./python/main.py  train --build ./{path}/build.app --weights-path ./weightsdir --agent ddpg --mem-path ./memdir
    
loading memory buffer from previous run:

    python ./python/main.py  train --build ./{path}/build.app --weights-path ./weightsdir --agent ddpg --mem-path ./memdir --load-mem

optional args:
    
    python ./python/main.py  train --build ./{path}/build.app --weights-path ./weightsdir --agent ddpg --mem-path ./memdir --print-agent-loss --num-obstacles 8 --num-agents 5



to test:

run main.py test -h to see options

example:

python ./python/main.py  test --build ./{path}/build.app --weights-path ./weightsdir --agent ddpg

### Other instructions:

Our project consists of 2 parts – the Unity game, and the python project.

If you want to just run/play with our python code, implement new agents etc., you can use the provided executables in the repo (windows, linux, and Mac). If you want to change more (reward systems, observations, other in-game changes) – download Unity, modify our game, and build your own version.

All software versions specified are the version we used when we created this project. These are the versions we know for sure to work with each other. You can install different versions, but you'll have to make sure they work together.

**Python part –**
 - In order to run our code please install **python 3.6.8**. this was the latest version to work with ml-agents repository when we created this project. Requested libraries are: 
-- pytorch 1.1 or higher (and all dependencies) 
-- mlagents 0.7

## Modifying Unity Build 

the following steps are to build easily our Unity project, corresponding to the following versions:

-- Unity **2018.3.7f1** 

-- Ml-agents release 0.7.0 (28.02.2019)


If you want to build latest version of ml-agents, refer to instructions in [https://github.com/Unity-Technologies/ml-agents](https://github.com/Unity-Technologies/ml-agents). Otherwise, the ml-agents unity files are already inside our repo, and no need to clone ml-agents repo at all.

- Download Unity:
-- Download and install Unity. You can download it from [https://unity.com/](https://unity.com/). The version of unity that this project was created with, is **2018.3.7f1.** any other version is not promised to work.
-- If working on windows, and you want to modify the scripts in unity, it is recommended to download _Microsoft Visual Studio Tool for Unity._ This tool made the C# scripting a lot easier for us.
- Setup the build (open our project):
-- Open a new project in unity
-- A folder "Assets" was created for this new project. Replace it (or create a soft link) with the "Assets" folder from our repo (Base\UnityEnvs\Assets).
-- Inside the Unity editor open our scene "Assets/ML-Agents/Examples/MyRace/MyRace.unity"
-- Build the project: in file/build_settings select the target platform, press "Add Open Scenes" and press Build.

