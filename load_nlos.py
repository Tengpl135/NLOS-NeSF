import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    
    # center = torch.Tensor(np.array([[1,0,0,0.45],[0,1,0,-1.88],[0,0,1,0.7],[0,0,0,1]])).float()
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    # c2w = center @ c2w 

    return c2w


def load_nlos_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'train']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_location = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        # import random
        # random.shuffle(meta['lights'])
        imgs = []
        locations = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = 1
        
        for light in meta['lights'][::skip]:
            fname = os.path.join(basedir, light['file_path'] + '.png')
            # fname = os.path.join(basedir, frame['file_path'])
            imgs.append(imageio.imread(fname))
            locations.append(np.array(light['location']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        locations = np.array(locations).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_location.append(locations)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(2)]
    
    imgs = np.concatenate(all_imgs, 0)
    location = np.concatenate(all_location, 0)
    
    # with open(os.path.join(basedir, 'camera_path.json'), 'r') as fp:
    #     metas_path = json.load(fp)
    # poses_path = []
    # for frame in metas_path['frames']:
    #     poses_path.append(np.array(frame['transform_matrix']))
    # poses_path = np.array(poses_path).astype(np.float32)
    
    
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    wall_location = list(meta['wall_location'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//8
        W = W//8
        focal = focal/8.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    # render_poses = poses_path
    return imgs, location, render_poses, [H, W, focal], wall_location, i_split


