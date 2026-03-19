
##############################################################################################

## Kod för att simulera BVR och för att visualisera. 

##############################################################################################

##############################################################################################

## Imports

## 

import jax
import jax.numpy as jnp
import numpy as np
import time

from black_box import classes
from black_box import game_env
from help_fcns import load_config_from_yaml
from black_box.display import plot_game_gif_gt
from mcts import Node

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
print(jax.devices())

############################################################################################### 

## Initiate

##

# Make changes in the .yaml file
Env = load_config_from_yaml()

Env.init_max_radius()
gamestate = Env.reset(1)
# batch size är satt till 1, kan ej ändras just nu

step_jit = jax.jit(Env.step)

################################################################################################

## Run trajectory

## 
terminal = False
stop = 10
N = 1
root = Node(gamestate, step_jit)

MCTS_traj = [gamestate]

#for _ in range(stop): # i = 0
while not terminal:

    # först kollar vi vem som ska agera
    next_time_planes = np.concatenate((gamestate.time_to_next_action_blue_plane, gamestate.time_to_next_action_red_plane), axis = 0)
    next_time_robots = np.concatenate((gamestate.time_to_next_action_blue_robots[0], gamestate.time_to_next_action_red_robots[0]), axis = 0)
    #if min(next_time_planes) <= min(next_time_robots):
    if next_time_planes[0] <= next_time_planes[1]:
        print("Blåa planet agerar")
        root.setTeam(gamestate.BLUE_plane.team)
        move = root.runSearch(10, gamestate.BLUE_plane.team)
        BLUE_action = jnp.array([move])
        RED_action = jnp.array([-1])
    else:
        print("Röda planet agerar")
        root.setTeam(gamestate.RED_plane.team)
        move = root.runSearch(10, gamestate.RED_plane.team)
        BLUE_action = jnp.array([-1])
        RED_action = jnp.array([move])
    #else:
    if min(next_time_planes) <= min(next_time_robots):
        print("Robot gör sig deterministiskt")

    # Här väljer vi handling: 0, 1, 2, 3, 4, eller 5.
    # 0: up left no shot, 1: forward no shot, 2: up right no shot,
    # 3: up left and shoot, 4: forward and shoot, 5: up right and shoot. 
    #BLUE_action = jnp.array([1])
    #RED_action = jnp.array([1])

    # Den här informationen kan du få av gamestate.
    # gamestate.batch_size  : batch storlek
    # gamestate.time        : tid                     
    # gamestate.steps       : hur mång steg som tagits 
    # gamestate.BLUE_plane  : Innehåller info om blåa planet
    # gamestate.RED_plane   : Plane                             
    # gamestate.BLUE_robots : Robots
    # gamestate.RED_robots  : Robots                            
    # gamestate.all_done    : 1 om alla spel är klara                      
    # gamestate.done            : 1 för de spel som är klara                  
    # gamestate.steps_effective : -1 tills spelet slut, då den sätt till spelets antal steg   
    # gamestate.result          : spelets resultat
    # gamestate.time_to_next_action_blue_plane  : tid till blå planet agerar
    # gamestate.time_to_next_action_red_plane   : tid till röd planet agerar
    # gamestate.time_to_next_action_blue_robots : tid till blåa robotar agerar
    # gamestate.time_to_next_action_red_robots  : tid till röda robotar agerar

    # Den här infomation kan du få av BLUE_plane och RED_plane.
    # time              : tid för senaste handling
    # team              : lag 
    # alive             : om vid liv
    # position          : position
    # direction         : riktning
    # shots_fired       : antal skot skjutna 
    # nose_radar_hit    : 1 om motsåndare träffad med nosradar   

    # Den här infomation kan du få av BLUE_robots och RED_robots.
    # time          : tid för senaste handling
    # team          : lag
    # active        : aktiv
    # position      : position
    # direction     : riktning
    # steps_taken   : antal steg
    
    current_time = time.time()
    # Med detta kan vi uppdatera gamestate
    gamestate = step_jit(gamestate, BLUE_action, RED_action)[0]

    # Startar om N nya spel från gamestate
    # Onödigt här, men kan behövas senare
    # N = 1
    # gamestate = Env.reset_to_state(gamestate, N)    
    time_after_step = time.time()
    print("This step took " + f"{time_after_step - current_time:.6f}" + " seconds.")

    # Vi sparar i MCTS_traj listan
    MCTS_traj.append(gamestate)
    newRoot = root.promoteToRoot(move)
    root = newRoot
    root.setState(gamestate)
    terminal = gamestate.all_done == 1

##

## Display trajectory as GIF

plot_game_gif_gt(Env, MCTS_traj, 'traj.gif')

## 