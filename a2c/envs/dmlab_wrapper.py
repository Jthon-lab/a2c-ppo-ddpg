import sys
import numpy as np
import deepmind_lab as deepmind_lab
import six
import cv2
from gym.spaces import Discrete,Box
def _action(*entries):
    return np.array(entries,dtype=np.intc)
ACTIONS = {
    'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
    'look_right': _action(20, 0, 0, 0, 0, 0, 0),
    'forward': _action(0, 0, 0, 1, 0, 0, 0),
}
VALID_ACTIONS = [np.array(_action(-20,0,0,0,0,0,0),dtype=np.intc),
                np.array(_action(20,0,0,0,0,0,0),dtype=np.intc),
                np.array(_action(0,0,0,1,0,0,0),dtype=np.intc)]
LEVEL_SENSORS = ['RGBD_INTERLEAVED',
                 'DEBUG.PLAYERS.EYE.POS',
                 'DEBUG.CAMERA_INTERLEAVED.TOP_DOWN',
                 'DEBUG.PLAYERS.VELOCITY',
                 'DEBUG.POS.ROT']
LEVEL_CONFIGS = {'width':'256',
                 'height':'256',
                 'fps':'60'}
class DMLab(object):
    def __init__(self,env_name,env_seed,timelimit,map_width=500,map_height=500):
        self.env_name = env_name
        self.env_seed = env_seed
        self.env_timelimit = timelimit
        self.map_width = map_width
        self.map_height = map_height
        self.env = deepmind_lab.Lab(self.env_name,LEVEL_SENSORS,LEVEL_CONFIGS,renderer='hardware')
        self.observation_space = Box(0,1,shape=(64,64,12))
        self.action_space = Discrete(3)
    def reset(self):
        self.env.reset(self.env_seed)
        self.env.step(np.array(_action(0,0,0,0,0,0,0),dtype=np.intc),num_steps=4)
        obs = self.env.observations()
        self.last_observation = obs

        self.rgbd = obs[LEVEL_SENSORS[0]]
        self.rgb = self.rgbd[:,:,0:3]
        self.norm_rgb = self._normalize_rgb_(self.rgb)
        self.depth = self.rgbd[:,:,3:4]
        self.norm_depth = self._normalize_depth_(self.depth)
        self.pos = obs[LEVEL_SENSORS[1]][0][0:2]
        self.pos[0] = self.pos[0] - 100
        self.pos[1] = self.map_height - self.pos[1] + 100
        self.norm_pos = self._normalize_pos_(self.pos)
        self.topdown = obs[LEVEL_SENSORS[2]]
        self.velocity = obs[LEVEL_SENSORS[3]]
        self.rotation = obs[LEVEL_SENSORS[4]]
        
        info = {}
        info['rgb'] = self.rgb
        info['norm_rgb'] = self.norm_rgb
        info['depth'] = self.depth
        info['norm_depth'] = self.norm_depth
        info['topdown'] = self.topdown
        info['pos'] = self.pos
        info['norm_pos'] = self.norm_pos
        info['velocity'] = self.velocity
        info['rotation'] = self.rotation
        info['map_width'] = self.map_width
        info['map_height'] = self.map_height

        self.experience_time = 0

        self.stack_rgb = np.concatenate((self.norm_rgb,self.norm_rgb,self.norm_rgb,self.norm_rgb),axis=2)
        return np.array(self.stack_rgb)#np.array(self.norm_rgb)#,info

    def step(self,action):
        assert action>=0 and action<3
        reward = self.env.step(VALID_ACTIONS[action],num_steps=4)
        done = not self.env.is_running()
        if not done:
            obs = self.env.observations()
        else:
            obs = self.last_observation
        self.last_observation = obs

        self.rgbd = obs[LEVEL_SENSORS[0]]
        self.rgb = self.rgbd[:,:,0:3]
        self.norm_rgb = self._normalize_rgb_(self.rgb)
        self.depth = self.rgbd[:,:,3:4]
        self.norm_depth = self._normalize_depth_(self.depth)
        self.pos = obs[LEVEL_SENSORS[1]][0][0:2]
        self.pos[0] = self.pos[0] - 100
        self.pos[1] = self.map_height - self.pos[1] + 100
        self.norm_pos = self._normalize_pos_(self.pos)

        self.topdown = obs[LEVEL_SENSORS[2]]
        self.velocity = obs[LEVEL_SENSORS[3]]
        self.rotation = obs[LEVEL_SENSORS[4]]

        info = {}
        info['rgb'] = self.rgb
        info['norm_rgb'] = self.norm_rgb
        info['depth'] = self.depth
        info['norm_depth'] = self.norm_depth
        info['topdown'] = self.topdown
        info['pos'] = self.pos
        info['norm_pos'] = self.norm_pos

        info['velocity'] = self.velocity
        info['rotation'] = self.rotation
        info['map_width'] = self.map_width
        info['map_height'] = self.map_height
        self.experience_time += 1

        if self.experience_time > self.env_timelimit or reward==10:
            done = True
        if reward != 10:
            reward = -0.02
        
        self.stack_rgb[:,:,0:3] = self.stack_rgb[:,:,3:6]
        self.stack_rgb[:,:,3:6] = self.stack_rgb[:,:,6:9]
        self.stack_rgb[:,:,6:9] = self.stack_rgb[:,:,9:12]
        self.stack_rgb[:,:,9:12] = self.norm_rgb
        return np.array(self.stack_rgb),reward,done,info

    def get_rgb(self):
        rgb = cv2.cvtColor(self.rgb,cv2.COLOR_BGR2RGB)
        return rgb
    def get_depth(self):
        return self.depth    
    def get_pos(self):
        return self.pos    
    def get_map_size(self):
        return np.array((self.map_width,self.map_height),np.float32)
    def get_topdown(self):
        return self.topdown
    def get_velocity(self):
        return self.velocity
    def get_rotation(self):
        return self.rotation
    
    def _normalize_depth_(self,depth):
        depth = cv2.resize(depth,(64,64))
        return np.array(depth/255.0,np.float32)
    def _normalize_rgb_(self,rgb):
        rgb = cv2.resize(rgb,(64,64))
        return np.array(rgb/127.5-1,np.float32)
    def _normalize_pos_(self,pos):
        x = pos[0]/self.map_width
        y = pos[1]/self.map_height
        return np.array((x,y),np.float32)
