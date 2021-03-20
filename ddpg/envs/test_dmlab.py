import numpy as np
import os
import cv2
import argparse
from dmlab_wrapper import DMLab

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name",type=str,default="fourrooms_maze_rl")
    parser.add_argument("--env_seed",type=int,default=2000)
    parser.add_argument("--timelimit",type=int,default=500)
    parser.add_argument("--map_width",type=int,default=500)
    parser.add_argument("--map_height",type=int,default=500)
    args = parser.parse_known_args()[0]
    return args

args = get_args()
env = DMLab(args.env_name,args.env_seed,args.timelimit,args.map_width,args.map_height)
obs = env.reset()
done = False
while not done:
    while True:
        key = cv2.waitKey(5)
        if key==97:
            act = 0
            break
        elif key==100:
            act = 1
            break
        elif key == 119:
            act = 2
            break
        cv2.imshow("obs",obs)
    
    next_obs,reward,done,info = env.step(act)
    print(reward)
    obs = next_obs