# Deep-Q playing Snake
This repository is inspired by Patrick Loeber (https://github.com/patrickloeber) and explores the usage of reinforcement learning applied to the game of Snake.



# Ideas for further improvement
Usually the model stops improving around 30-40 score, at this point the snake usually traps itself and has nowhere to go. The reason for this is that the snake is that the agent is not aware of the snakes location except for the head. A potential improvement would therefore be to include the snake's body in state, or, to use a model that allows the agent to have some memory of its previous moves to avoid trapping itself.