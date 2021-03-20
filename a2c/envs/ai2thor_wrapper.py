from ai2thor.controller import Controller
import cv2
import numpy as np
from gym.spaces import Box,Discrete
ENV_LIST=['FloorPlan_Train1_1', 'FloorPlan_Train1_2', 'FloorPlan_Train1_3', 'FloorPlan_Train1_4', 'FloorPlan_Train1_5',
          'FloorPlan_Train2_1', 'FloorPlan_Train2_2', 'FloorPlan_Train2_3', 'FloorPlan_Train2_4', 'FloorPlan_Train2_5',
          'FloorPlan_Train3_1', 'FloorPlan_Train3_2', 'FloorPlan_Train3_3', 'FloorPlan_Train3_4', 'FloorPlan_Train3_5',
          'FloorPlan_Train4_1', 'FloorPlan_Train4_2', 'FloorPlan_Train4_3', 'FloorPlan_Train4_4', 'FloorPlan_Train4_5',
          'FloorPlan_Train5_1', 'FloorPlan_Train5_2', 'FloorPlan_Train5_3', 'FloorPlan_Train5_4', 'FloorPlan_Train5_5']


#TODO Convert segmentation result and detection result into matrix
#TODO Add Reward Function for this environment

class AI2THOR(object):
    def __init__(self,env_name,timelimit,stack_frame=1,color=True,image_size=128,render_all=True):
        assert env_name in ENV_LIST
        self.env_name = env_name
        self.timelimit = timelimit
        if render_all == True:
            self.controller = Controller(scene=self.env_name, agentMode='bot',gridSize=0.25,rotateStepDegrees=10,renderDepthImage=True,renderObjectImage=True,renderClassImage=True)
        else:
            self.controller = Controller(scene=self.env_name, agentMode='bot',gridSize=0.25,rotateStepDegrees=10)
        
        self.time = 0
        self.score = 0

        self.color = color
        self.stack_frame = stack_frame
        self.image_size =image_size
        self.render_all = render_all
        self.action_space = Discrete(5)
        if color==True:
            self.observation_space = Box(-1,1,(image_size,image_size,3*stack_frame,))
            self.channel = 3
        else:
            self.observation_space = Box(-1,1,(image_size,image_size,1*stack_frame,))
            self.channel = 1
        self.state = np.zeros(self.observation_space.shape,dtype=np.float32)

    def __normalize__(self,image):
        image = np.array(image/127.5-1,np.float32)
        return image
    def __denormalize__(self,image):
        image = np.array((image+1)*127.5,np.uint8)
        return image

    def reset(self):
        self.score = 0
        self.controller.reset(scene=self.env_name)
        event = self._stay()

        obs = event.frame
        obs = cv2.cvtColor(obs,cv2.COLOR_BGR2RGB)
        self.rgb = np.array(obs)
        obs = cv2.resize(obs,(self.image_size,self.image_size))
        if self.color == False:
            obs = cv2.cvtColor(obs,cv2.COLOR_BGR2GRAY)
            obs = np.expand_dims(obs,-1)
        obs = self.__normalize__(obs)

        self.state[:,:,0:self.channel] = obs
        self.time = 0
        return np.array(self.state)

    def render(self):
        return self.rgb

    def step(self,act):
        assert act>=0 and act<5
        if act==0:
            event = self._moveahead()
        elif act==1:
            event = self._moveback()
        elif act==2:
            event = self._rotateleft()
        elif act==3:
            event = self._rotateright()
        elif act==4:
            event = self._stay()

        next_obs = event.frame
        next_obs = cv2.cvtColor(next_obs,cv2.COLOR_BGR2RGB)
        self.rgb = np.array(next_obs)
        next_obs = cv2.resize(next_obs,(self.image_size,self.image_size))
        if self.color == False:
            next_obs = cv2.cvtColor(next_obs,cv2.COLOR_BGR2GRAY)
            next_obs = np.expand_dims(next_obs,-1)
        next_obs = self.__normalize__(next_obs)

        for i in range(1,self.stack_frame):
            self.state[:,:,self.channel*(self.stack_frame-i):self.channel*(self.stack_frame-i+1)] = self.state[:,:,self.channel*(self.stack_frame-i-1):self.channel*(self.stack_frame-i)]
        self.state[:,:,0:self.channel] = next_obs

        self.time+=1
        done = (self.time>=self.timelimit)
        #TODO Reward Function needs to be implemented
        reward = self._reward_func()

        if self.time>self.timelimit:
            print("Warning: Exceed the permitted time limit")
        if self.render_all == True:
            info = self.wrap_metadata(event)
        else:
            info = {}
        self.score += reward
        return np.array(self.state),reward,done,info
    def draw_bbox(self,image,color_dict,bbox_dict):
        draw_image = np.array(image,np.uint8)
        for key in bbox_dict.keys():
            rgb = color_dict[key]
            bbox = bbox_dict[key]
            startx = int(bbox[0]) 
            starty = int(bbox[1])
            endx = int(bbox[2])
            endy = int(bbox[3])
            cv2.rectangle(draw_image,(startx,starty),(endx,endy),color=rgb,thickness=2)
        return draw_image
    
    def wrap_metadata(self,event):
        draw_image = event.frame
        depth = event.depth_frame
        depth = depth/5.0
        
        id_color_dict = event.color_to_object_id
        color_id_dict = event.object_id_to_color
        
        instance_seg = event.instance_segmentation_frame
        class_seg = event.class_segmentation_frame
        
        instance_det = event.instance_detections2D
        instance_det_vis = self.draw_bbox(draw_image,color_id_dict,instance_det)
        
        class_det = event.class_detections2D
        del_keys = []
        for key in class_det:
            class_det[key] = class_det[key][0]
            if key not in color_id_dict.keys():
                del_keys.append(key)
        for key in del_keys:
            del[class_det[key]]
       
        class_det_vis = self.draw_bbox(draw_image,color_id_dict,class_det)
        info = {'depth':depth,
                'instance_segmentation':instance_seg,
                'class_segmentation':class_seg,
                'instance_detection':instance_det_vis,
                'class_detection':class_det_vis}
        return info


    def _stay(self):
        event = self.controller.step('GetReachablePositions')
        return event
    def _moveahead(self):
        event = self.controller.step('MoveAhead')
        return event
    def _moveback(self):
        event = self.controller.step('MoveBack')
        return event
    def _rotateleft(self):
        event = self.controller.step('RotateLeft')
        return event
    def _rotateright(self):
        event = self.controller.step('RotateRight')
        return event
    
    #TODO Reward Function : based on the next_state and goal?
    def _reward_func(self):
        reward = 0
        return reward

'''
#Test Wrapper
env = AI2THOR("FloorPlan_Train3_1",100,3,True,128,True)
state = env.reset()
done = False
time = 0
while not done:
    while True:
        key = cv2.waitKey(5)
        if key==97:
            act = 2
            break
        elif key==100:
            act = 3
            break
        elif key==115:
            act = 1
            break
        elif key==119:
            act = 0
            break
        elif key==27:
            act = 4
            break
        cv2.imshow("rgb",env.render())
        if time!=0:
           cv2.imshow("depth",depth)
           cv2.imshow("int_seg",instance_seg) 
           cv2.imshow("class_seg",class_seg)
           cv2.imshow("int_det",instance_det) 
           cv2.imshow("class_det",class_det) 
    next_state,reward,done,info = env.step(act)

    depth = info['depth']
    instance_seg = info['instance_segmentation']
    class_seg = info['class_segmentation']
    instance_det = info['instance_detection']
    class_det = info['class_detection']
    time = 1
    state = next_state'''
