import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import cv2

import matplotlib.pyplot as plt

from run_nerf_helpers import *


from load_nlos import load_nlos_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

def to_rgb(imgs):
    img_list = []
    for i in range(len(imgs)):
        bgr = cv2.applyColorMap((cv2.normalize(imgs[i], None, 0, 1, cv2.NORM_MINMAX)*255).astype(np.uint8), cv2.COLORMAP_JET)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_list.append(rgb)
    return img_list


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    magnitude_d = torch.norm(rays_d / rays_d[:,0:1], dim=-1, keepdim=True)
    
    # near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    D = 10
    near = 0.3
    near, far = near * torch.ones_like(rays_d[...,:1]), D * magnitude_d
    
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'trans_map', 'disp_map', 'acc_map', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf
    print(f"render test(H,W):{H}, {W} ")

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor
    rgbs = []
    trans_map = []
    disps = []
    depths = []
    ro = 3*np.pi/4
    rl = np.pi/2
    if savedir is not None:
        num = 1
    else:
        num = 90
        
    dr = rl / num
    import time
    
    for i in range(num):
        print(f"test num: {i} total: {num}")
        rz = ro + dr*i
        # rays_o, rays_d = get_rays_euler(H,W, focal, [0., 0.,rz-np.pi], [3.5*np.cos(rz), 3.5*np.sin(rz), -0.3])
        # rays_o, rays_d = get_rays_euler(H,W, focal, [0., 0., 0.], [-3.5, 0.0, -0.3])
        rays_o, rays_d = get_rays_euler(H,W, focal, [0., 0., 0.], [-5, 0.0, 0])
        # print(f"rays_d {rays_d.shape}")
        rays = [torch.Tensor(rays_o).to(device),torch.Tensor(rays_d).to(device)]
        
        start_time = time.time()
        rgb, trans, disp, acc, depth, _ = render(H, W, K, chunk=chunk, rays=rays, **render_kwargs)
        rgb = rgb.transpose(0,1).squeeze(-1)
        disp = disp.transpose(0,1)
        trans = trans.transpose(0,1)
        depth = depth.transpose(0,1)
        
        
        
        # depth[depth<10] = 40
        # depth = 10 - depth
        # print(f"rgb {rgb.shape}")
        # print(f"rgb.max {torch.max(rgb)}")
        rgbs.append(rgb.cpu().numpy())
        trans_map.append(trans.cpu().numpy())
        disps.append(disp.cpu().numpy())
        depths.append(depth.cpu().numpy())
        
        
        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """
        def normalize_background(img):
            img[img == 0] = 255
            return img
            
        
        
        if savedir is not None:
            # rgb8 = to8b(rgbs[-1])
            # disp8 = to8b(disps[-1])
            # trans8 = to8b(trans_map[-1])
            

            disp_rgb = cv2.applyColorMap((cv2.normalize(disps[-1], None, 0, 1, cv2.NORM_MINMAX)*255).astype(np.uint8), cv2.COLORMAP_HOT)
            trans_rgb = cv2.applyColorMap((cv2.normalize(trans_map[-1], None, 0, 1, cv2.NORM_MINMAX)*255).astype(np.uint8), cv2.COLORMAP_HOT)
            depth_rgb = cv2.applyColorMap((cv2.normalize(depths[-1], None, 0, 1, cv2.NORM_MINMAX)*255).astype(np.uint8), cv2.COLORMAP_HOT)
            # depth_rgb = cv2.applyColorMap((depths[-1]/30*255).astype(np.uint8), cv2.COLORMAP_HOT)
            depth_rgb_0 = (cv2.normalize(depths[-1], None, 0, 1, cv2.NORM_MINMAX)*255).astype(np.uint8)
            # depth_rgb_0 = (depths[-1]/30*255).astype(np.uint8).astype(np.uint8)
            
            # imageio.imwrite(filename, rgb8)
            filename = os.path.join(savedir, 'dis_{:03d}.png'.format(i))
            cv2.imwrite(filename,disp_rgb)
            # imageio.imwrite(filename, disp_rgb)
            filename = os.path.join(savedir, 'trans_{:03d}.png'.format(i))
            cv2.imwrite(filename,trans_rgb)
            # imageio.imwrite(filename, trans_rgb)
            
            filename = os.path.join(savedir, 'depth_{:03d}.png'.format(i))
            cv2.imwrite(filename,depth_rgb)
            
            filename = os.path.join(savedir, 'depth0_{:03d}.png'.format(i))
            cv2.imwrite(filename,depth_rgb_0)
            filename = os.path.join(savedir, 'depth0_{:03d}.npy'.format(i))
            np.save(filename, depths[-1])
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            all_time = end_time - s_time - elapsed_time

            print(f"渲染耗时: {elapsed_time} s")
            print(f"总耗时: {all_time} s")
            print(f"image_index:{savedir} ")

    rgbs = np.stack(rgbs, 0)    # [f_num, h, w, c]
    disps = np.stack(disps, 0)
    trans_map = np.stack(trans_map,0)
    depths = np.stack(depths,0)
    

    return rgbs, disps, trans_map, depths

def render_lights(hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, locations=None, wall_location=None, scale=1.):

    H, W, focal = hwf
    print(f"render test light (H,W):{H}, {W} ")

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor
    rgbs = []
    trans_map = []
    disps = []
    depths = []

    for i,location in enumerate(locations):
        print(f"test num: {i} total: {len(locations)}")
        
        rays_o, rays_d = get_rays_np_wall(H, W, focal, location, wall_location, scale=scale)
        # print(f"rays_d {rays_d.shape}")
        rays = [torch.Tensor(rays_o).to(device),torch.Tensor(rays_d).to(device)]
        
        rgb, trans, disp, acc, depth, _ = render(H, W, K, chunk=chunk, rays=rays, **render_kwargs)
        # rgb = rgb.transpose(0,1).squeeze(-1)
        # disp = disp.transpose(0,1)
        # trans = trans.transpose(0,1)
        # depth = depth.transpose(0,1)
        
        # depth[depth<0.5] = 8
        # print(f"rgb {rgb.shape}")
        # print(f"rgb.max {torch.max(rgb)}")
        rgbs.append(rgb.cpu().numpy())
        trans_map.append(trans.cpu().numpy())
        disps.append(disp.cpu().numpy())
        depths.append(depth.cpu().numpy())

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """
        def normalize_background(img):
            img[img == 0] = 255
            return img
            
        
        
        if savedir is not None:
            # rgb8 = to8b(rgbs[-1])
            # disp8 = to8b(disps[-1])
            # trans8 = to8b(trans_map[-1])
            
            rgb_rgb = cv2.applyColorMap((cv2.normalize(rgbs[-1], None, 0, 1, cv2.NORM_MINMAX)*255).astype(np.uint8), cv2.COLORMAP_JET)
            disp_rgb = cv2.applyColorMap((cv2.normalize(disps[-1], None, 0, 1, cv2.NORM_MINMAX)*255).astype(np.uint8), cv2.COLORMAP_JET)
            trans_rgb = cv2.applyColorMap((cv2.normalize(trans_map[-1], None, 0, 1, cv2.NORM_MINMAX)*255).astype(np.uint8), cv2.COLORMAP_JET)
            depth_rgb = cv2.applyColorMap((cv2.normalize(depths[-1], None, 0, 1, cv2.NORM_MINMAX)*255).astype(np.uint8), cv2.COLORMAP_JET)
            depth_rgb_0 = (cv2.normalize(depths[-1], None, 0, 1, cv2.NORM_MINMAX)*255).astype(np.uint8)
            
            filename = os.path.join(savedir, 'rgb_{:03d}.png'.format(i))
            cv2.imwrite(filename,rgb_rgb)
            # imageio.imwrite(filename, rgb8)
            filename = os.path.join(savedir, 'dis_{:03d}.png'.format(i))
            cv2.imwrite(filename,disp_rgb)
            # imageio.imwrite(filename, disp_rgb)
            filename = os.path.join(savedir, 'trans_{:03d}.png'.format(i))
            cv2.imwrite(filename,trans_rgb)
            # imageio.imwrite(filename, trans_rgb)
            
            filename = os.path.join(savedir, 'depth_{:03d}.png'.format(i))
            cv2.imwrite(filename,depth_rgb)
            
            filename = os.path.join(savedir, 'rays_o_{:03d}.npy'.format(i))
            np.save(filename,rays_o)
            
            filename = os.path.join(savedir, 'rays_d_{:03d}.npy'.format(i))
            np.save(filename,rays_d)
            
            filename = os.path.join(savedir, 'depth0_{:03d}.npy'.format(i))
            np.save(filename, depths[-1])


    rgbs = np.stack(rgbs, 0)    # [f_num, h, w, c]
    disps = np.stack(disps, 0)
    trans_map = np.stack(trans_map,0)
    depths = np.stack(depths,0)
    

    return rgbs, disps, trans_map, depths


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)

    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:,:-1]
    
    trans = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    trans_map = trans[:,-1]
    # rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    rgb_map = torch.sum(weights[...,None], -2)  # [N_rays, 3]
    weight_map = torch.sum(weights, -1)
    # print("weight_map.shape ",weight_map.shape)

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    # print(f"disp_map max: {torch.max(disp_map)} min: {torch.min(disp_map)}")
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, trans_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
        # print(z_vals.shape)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, trans_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, trans_map_0, disp_map_0, acc_map_0 = rgb_map, trans_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, trans_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'trans_map' : trans_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map': depth_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['trans0'] = trans_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, default='configs/test.txt',
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--render_observe", action='store_true')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=50, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=5000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')
    parser.add_argument("--if_gray", action='store_true', 
                        help='set for spherical 360 scenes')

    return parser

def divide_and_mark_regions(images, rays, num_partitions=4, dense_factor=2, sparse_factor=4):
    """ 将图像分为多个区域，并标记出稀疏、密集和边界区域 """
    contour_images_high = []  # 存储被采样区域的图像
    contour_images_low = []  # 存储被采样区域的图像
    contour_rays_high = []  # 存储对应的高光线区域数据
    contour_rays_low = []  # 存储对应的低光线区域数据
    region_images = []  

    for n in range(images.shape[0]):
        current_image = images[n]
        current_rays = rays[n]  # 获取对应的光线数据
        gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        
        # 使用Sobel算法计算边缘
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobel_x, sobel_y)
        
        height, width = edges.shape
        partition_height = height // num_partitions
        partition_width = width // num_partitions
        marked_image = current_image.copy()
        
        mean_edge_strength = np.mean(edges)
        high_threshold = mean_edge_strength * 500

        for i in range(num_partitions):
            for j in range(num_partitions):
                y_start = i * partition_height
                y_end = (i + 1) * partition_height
                x_start = j * partition_width
                x_end = (j + 1) * partition_width
                
                region_edges = edges[y_start:y_end, x_start:x_end]
                edge_strength = np.sum(region_edges)
                
                if edge_strength > high_threshold:  # 密集区域
                    cv2.rectangle(marked_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                    # 对密集区域进行降采样为原来的二分之一（1/4)
                    dense_region = current_image[y_start:y_end, x_start:x_end]
                    dense_region_downsampled = dense_region[::dense_factor, ::dense_factor]
                    contour_images_high.append(dense_region_downsampled)

                    # 获取对应的光线数据
                    dense_rays = current_rays[:, y_start:y_end, x_start:x_end]  # 形状为 (2, partition_height, partition_width, 3)
                    dense_rays_downsampled = dense_rays[:, ::dense_factor, ::dense_factor]  # 对应的降采样
                    contour_rays_high.append(dense_rays_downsampled)

                else:  # 稀疏区域
                    cv2.rectangle(marked_image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
                    # 对稀疏区域进行降采样到原来的四分之一(1/16)
                    sparse_region = current_image[y_start:y_end, x_start:x_end]
                    sparse_region_downsampled = sparse_region[::sparse_factor, ::sparse_factor]
                    contour_images_low.append(sparse_region_downsampled)

                    # 获取对应的光线数据
                    sparse_rays = current_rays[:, y_start:y_end, x_start:x_end]  # 形状为 (2, partition_height, partition_width, 3)
                    sparse_rays_downsampled = sparse_rays[:, ::sparse_factor, ::sparse_factor]  # 对应的降采样
                    contour_rays_low.append(sparse_rays_downsampled)

        region_images.append(marked_image)
    print(f"region: {num_partitions*num_partitions}, dense_factor: {dense_factor}, sparse_factor: {sparse_factor}")
    return np.array(contour_images_high), np.array(contour_images_low), np.array(contour_rays_high), np.array(contour_rays_low)

def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None

    images, locations, render_poses, hwf, wall_location, i_split = load_nlos_data(args.datadir, args.half_res, args.testskip)
    print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
    i_train, i_test = i_split

    near = 1.
    far = 10.

    images = images[...,:3]

    

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    # if args.render_test:
    #     render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        num = locations.shape[0] // 2
        images_region = images[num:]
        rays = np.stack([get_rays_np_wall(H, W, focal, location, wall_location, scale=1.) for location in locations[:num]], 0) # [N, ro+rd, H, W, 3]
        
        # 处理图像并标记区域 images(N, H, W, 3), rays(N, 2, H, W, 3)
        images_dense_region, images_sparse_region, rays_dense_region, rays_sparse_region = divide_and_mark_regions(images_region, rays, num_partitions=16, dense_factor=2, sparse_factor=100)
        print(f"rays_dense_region shape: {rays_dense_region.shape}")
        print(f"images_dense_region shape: {images_dense_region.shape}")

        # 构造密集区域光线
        rays_rgb_dense = np.concatenate([rays_dense_region, images_dense_region[:, None, :, :, :], images_dense_region[:, None, :, :, :]], 1) # [N, ro+rd+2, H, W, 3]
        rays_rgb_dense = np.transpose(rays_rgb_dense, [0,2,3,1,4]) # [N, H, W, ro+rd+2, 3]
        rays_rgb_dense = np.reshape(rays_rgb_dense, [-1,4,3]) # [N*H*W, ro+rd+2, 3]
        print(f"rays_rgb_dense: {rays_rgb_dense.shape}")  
        # 构造稀疏区域光线
        rays_rgb_sparse = np.concatenate([rays_sparse_region, images_sparse_region[:, None, :, :, :], images_sparse_region[:, None, :, :, :]], 1) # [N, ro+rd+2, H, W, 3]
        rays_rgb_sparse = np.transpose(rays_rgb_sparse, [0,2,3,1,4]) # [N, H, W, ro+rd+2, 3]
        rays_rgb_sparse = np.reshape(rays_rgb_sparse, [-1,4,3]) # [N*H*W, ro+rd+2, 3] 
        print(f"rays_rgb_sparse: {rays_rgb_sparse.shape}")  
        rays_rgb = np.concatenate([rays_rgb_dense, rays_rgb_sparse], 0) # (dense+sparse, 4, 3)
        print(f"rays_rgb: {rays_rgb.shape}")  
        rays_rgb = rays_rgb.astype(np.float32)
        
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    # poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)


    N_iters = 50000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    # print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s, rgb_s = batch[:2], batch[2], batch[3]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        # else:
        #     # Random from one image
        #     img_i = np.random.choice(i_train)
        #     target = images[img_i]
        #     target = torch.Tensor(target).to(device)
        #     # pose = poses[img_i, :3,:4]

        #     if N_rand is not None:
        #         # rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

        #         if i < args.precrop_iters:
        #             dH = int(H//2 * args.precrop_frac)
        #             dW = int(W//2 * args.precrop_frac)
        #             coords = torch.stack(
        #                 torch.meshgrid(
        #                     torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
        #                     torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
        #                 ), -1)
        #             if i == start:
        #                 print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
        #         else:
        #             coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

        #         coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
        #         select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
        #         select_coords = coords[select_inds].long()  # (N_rand, 2)
        #         rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        #         rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        #         batch_rays = torch.stack([rays_o, rays_d], 0)
        #         target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, trans, disp, acc, depth, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)
        
        target_trans_s = torch.mean(target_s, -1)
        rgb_mean_s = torch.mean(rgb_s, -1) * 1.0

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_trans_s)
        trans_loss = img2mse(1 - trans, target_trans_s)
        # trans = extras['raw'][...,-1]
        
        loss = trans_loss
        psnr = mse2psnr(loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_trans_s)
            trans_loss0 = img2mse(1 - extras['trans0'], target_trans_s)
            loss = loss + trans_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)
            
        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                render_path(hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            # with torch.no_grad():
            #     render_lights(hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir,locations=locations, wall_location=wall_location,scale=1.)
        if i%args.i_video==0 and i > 0:
            with torch.no_grad():
                rgbs, disps, trans, depths = render_path(hwf, K, args.chunk, render_kwargs_test)     # rgbs: [f_num, h, w, c]
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to_rgb(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to_rgb(disps), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'trans.mp4', to_rgb(trans), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'depth.mp4', depths, fps=30, quality=8)
            
    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    s_time = time.time()
    

    train()
