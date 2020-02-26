# RL_Multy_Agent_Unity

Open a new Unity Project ->
Replace The "Assets" folder in it with this one.

build the project (My Ball) ->
move build folder to python/ddpg ->
now you can the test script


once you have working unity build you can run the main.py
to train:

run main.py train -h to see options
example:

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
