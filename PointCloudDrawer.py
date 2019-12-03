# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

"""
OpenCV and Numpy Point cloud Software Renderer

This sample is mostly for demonstration and educational purposes.
It really doesn't offer the quality or performance that can be
achieved with hardware acceleration.

Usage:
------
Mouse: 
    Drag with left button to rotate around pivot (thick small axes), 
    with right button to translate and the wheel to zoom.

Keyboard: 
    [p]     Pause
    [r]     Reset View
    [d]     Cycle through decimation values
    [z]     Toggle point scaling
    [s]     Save PNG (./out.png)
    [e]     Export points to ply (./out.ply)
    [q\ESC] Quit
    
    
    [a]     press [a] when it is pause, start to display frame by frame
"""

import math
import cv2
import numpy as np
import pyrealsense2 as rs



class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = False
        self.color = True
        self.per_frame = False
        self.w = 0
        self.h = 0
        

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


    def get_size(self, out):
        self.h, self.w = out.shape[:2]

    # Here is to register the mouse event, be sure to copy this part to the main script #
    def mouse_cb(self, event, x, y, flags, param):
    
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_btns[0] = True
    
        if event == cv2.EVENT_LBUTTONUP:
            self.mouse_btns[0] = False
    
        if event == cv2.EVENT_RBUTTONDOWN:
            self.mouse_btns[1] = True
    
        if event == cv2.EVENT_RBUTTONUP:
            self.mouse_btns[1] = False
    
        if event == cv2.EVENT_MBUTTONDOWN:
            self.mouse_btns[2] = True
    
        if event == cv2.EVENT_MBUTTONUP:
            self.mouse_btns[2] = False
    
        if event == cv2.EVENT_MOUSEMOVE:
    

            h = self.h
            w = self.w
            dx, dy = x - self.prev_mouse[0], y - self.prev_mouse[1]
    
            if self.mouse_btns[0]:
                self.yaw += float(dx) / w * 2
                self.pitch -= float(dy) / h * 2
    
            elif self.mouse_btns[1]:
                dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
                self.translation -= np.dot(self.rotation, dp)
    
            elif self.mouse_btns[2]:
                dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
                self.translation[2] += dz
                self.distance -= dz
    
        if event == cv2.EVENT_MOUSEWHEEL:
            dz = math.copysign(0.1, flags)
            self.translation[2] += dz
            self.distance -= dz
    
        self.prev_mouse = (x, y)
    

   
    
    
    
    def project(self, out, v):
        """project 3d vector array to 2d"""
        h, w = out.shape[:2]
        view_aspect = float(h)/w
    
        # ignore divide by zero for invalid depth
        with np.errstate(divide='ignore', invalid='ignore'):
            proj = v[:, :-1] / v[:, -1, np.newaxis] * \
                (w*view_aspect, h) + (w/2.0, h/2.0)
    
        # near clipping
        znear = 0.03
        proj[v[:, 2] < znear] = np.nan
        return proj
    
    
    def view(self, v):
        """apply view transformation on vector array"""
        return np.dot(v - self.pivot, self.rotation) + self.pivot - self.translation
    

    
    def line3d(self, out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
        """draw a 3d line from pt1 to pt2"""
        #p0 = self.project(out, pt1)[0]
        p0 = self.project(out, pt1.reshape(-1, 3))[0]

        p1 = self.project(out, pt2.reshape(-1, 3))[0]
        if np.isnan(p0).any() or np.isnan(p1).any():
            return
        p0 = tuple(p0.astype(int))
        p1 = tuple(p1.astype(int))
        rect = (0, 0, out.shape[1], out.shape[0])   # image rectangle size
        inside, p0, p1 = cv2.clipLine(rect, p0, p1) # clip the line within the rect
        if inside:
            cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)
    
    
   

    
    
    def grid(self, out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
        """draw a grid on xz plane"""
        pos = np.array(pos)
        s = size / float(n)
        s2 = 0.5 * size
        for i in range(0, n+1):
            x = -s2 + i*s
            self.line3d(out, self.view(pos + np.dot((x, 0, -s2), self.rotation)),
                   self.view(pos + np.dot((x, 0, s2), self.rotation)), color)
        for i in range(0, n+1):
            z = -s2 + i*s
            self.line3d(out, self.view(pos + np.dot((-s2, 0, z), self.rotation)),
                   self.view(pos + np.dot((s2, 0, z), self.rotation)), color)
    
    
    def axes(self, out, pos, rotation=np.eye(3), size=0.075, thickness=2):
        """draw 3d axes"""
        self.line3d(out, pos, pos +
               np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
        self.line3d(out, pos, pos +
               np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
        self.line3d(out, pos, pos +
               np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)
    
    
    def frustum(self, out, intrinsics, color=(0x40, 0x40, 0x40)):
        """draw camera's frustum"""
        orig = self.view([0, 0, 0])
        w, h = intrinsics.width, intrinsics.height
    
        for d in range(1, 6, 2):
            def get_point(x, y):
                p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
                self.line3d(out, orig, self.view(p), color)
                return p
    
            top_left = get_point(0, 0)
            top_right = get_point(w, 0)
            bottom_right = get_point(w, h)
            bottom_left = get_point(0, h)
    
            self.line3d(out, self.view(top_left), self.view(top_right), color)
            self.line3d(out, self.view(top_right), self.view(bottom_right), color)
            self.line3d(out, self.view(bottom_right), self.view(bottom_left), color)
            self.line3d(out, self.view(bottom_left), self.view(top_left), color)
    
    
    
    
    def pointcloud_display(self, out, verts, texcoords, color, painter=True):
        """draw point cloud with optional painter's algorithm"""
        if painter:
            # Painter's algo, sort points from back to front
    
            # get reverse sorted indices by z (in view-space)
            # https://gist.github.com/stevenvo/e3dad127598842459b68
            v = self.view(verts)
            s = v[:, 2].argsort()[::-1]
            proj = self.project(out, v[s])
        else:
            proj = self.project(out, self.view(verts))
    
        if self.scale:
            proj *= 0.5**self.decimate
    
        h, w = out.shape[:2]
    
        # proj now contains 2d image coordinates
        j, i = proj.astype(np.uint32).T
    
        # create a mask to ignore out-of-bound indices
        im = (i >= 0) & (i < h)
        jm = (j >= 0) & (j < w)
        m = im & jm
        
        # Print the size of pointdata
    #    print("m: ", m.shape)
    
    
        cw, ch = color.shape[:2][::-1]
        if painter:
            # sort texcoord with same indices as above
            # texcoords are [0..1] and relative to top-left pixel corner,
            # multiply by size and add 0.5 to center
            v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
        else:
            v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
        # clip texcoords to image
        np.clip(u, 0, ch-1, out=u)
        np.clip(v, 0, cw-1, out=v)
    
        # perform uv-mapping
        out[i[m], j[m]] = color[u[m], v[m]]
    
        
        return m.size

