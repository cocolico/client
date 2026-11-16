"""Map local raw camera rays onto the server's raw cam0 pixel geometry.

This is an explicit adapter for a different calibrated camera, not an identity
claim about the training camera. It preserves optical axes and metric 3D points.
The target grid retains cam0 distortion; the server still does its normal
undistortion. No JPEG, changed extrinsic, or task-table transfer is involved.
"""
import cv2
import numpy as np


class Cam0Geometry:
    def __init__(self,source,target):
        self.source_size = (int(source['width']),int(source['height']))
        self.target_size = (int(target['width']),int(target['height']))
        x,y = np.meshgrid(np.arange(self.target_size[0]),np.arange(self.target_size[1]))
        pixels = np.stack([x,y],axis=-1).astype(np.float64).reshape(-1,1,2)
        rays = cv2.undistortPoints(pixels,np.asarray(target['K'],float).reshape(3,3),
                                    np.asarray(target['D'],float)).reshape(-1,2)
        points = np.c_[rays,np.ones(len(rays))]
        source_pixels,_ = cv2.projectPoints(points,np.zeros(3),np.zeros(3),
            np.asarray(source['K'],float).reshape(3,3),np.asarray(source['D'],float))
        maps = source_pixels.reshape(self.target_size[1],self.target_size[0],2)
        self.map_x = maps[...,0].astype(np.float32); self.map_y = maps[...,1].astype(np.float32)
        valid = (self.map_x>=0)&(self.map_x<=self.source_size[0]-1)&(self.map_y>=0)&(self.map_y<=self.source_size[1]-1)
        self.coverage = float(valid.mean())
        if self.coverage < .98: raise ValueError(f'Local camera covers only {self.coverage:.2%} of cam0 rays')

    def apply(self,rgb):
        if rgb.dtype != np.uint8 or rgb.shape != (self.source_size[1],self.source_size[0],3):
            raise ValueError('Raw image does not match calibrated source profile')
        return cv2.remap(rgb,self.map_x,self.map_y,cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,borderValue=0)
