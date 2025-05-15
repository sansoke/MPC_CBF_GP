#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

# author: Daniel Kloeser
import numpy as np
from tracks.readDataFcn import getTrack
import scipy.interpolate as interp
import math


def parseReference(x, y):
    
    if len(x) < 2:
        print("Reference path must have at least 2 points")
        return None, None, None, None, None, None
    
    x1, x2 = x[0], x[1]
    y1, y2 = y[0], y[1]
    dist_s = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    xe, xee = x[-1], x[-2]
    ye, yee = y[-1], y[-2]
    dist_e  = ((xe - xee) ** 2 + (ye - yee) ** 2) ** 0.5

    x_slope_start, x_slope_end = (x2 - x1) / dist_s, (xe - xee) / dist_e
    y_slope_start, y_slope_end = (y2 - y1) / dist_s, (ye - yee) / dist_e
    
    ds = [0]
    distance = 0
    
    for i in range(1, len(x)):
        
        x_now, y_now = x[i], y[i]
        x_prv, y_prv = x[i-1], y[i-1]
        distance = math.sqrt((x_now - x_prv)**2 + (y_now - y_prv)**2)
        ds.append(distance + ds[-1])

    x_spline = interp.CubicSpline(ds, x, bc_type=((1, x_slope_start), (1, x_slope_end)))
    y_spline = interp.CubicSpline(ds, y, bc_type=((1, y_slope_start), (1, y_slope_end)))

    density = 0.05
    dense_s = np.linspace(ds[0], ds[-1], int(ds[-1]/density)) # Change 1000 to the density you want

    # Get first derivatives
    dx_ds = x_spline.derivative()(dense_s)
    dy_ds = y_spline.derivative()(dense_s)

    # Get second derivatives
    dx2_ds2 = x_spline.derivative(nu=2)(dense_s)
    dy2_ds2 = y_spline.derivative(nu=2)(dense_s)

    # Compute phi (slope angle)
    phi = np.arctan2(dy_ds, dx_ds)

    # Compute kappa (curvature)
    kappa = (dx_ds * dy2_ds2 - dy_ds * dx2_ds2) / (dx_ds**2 + dy_ds**2)**(1.5)

    return x_spline, y_spline, ds[-1], dense_s, phi, kappa



def transformProj2Orig(si, ni, alpha, v, filename='LMS_Track.txt'):
    [sref, xref, yref, psiref, _] = getTrack(filename=filename)
    tracklength = sref[-1]
    si = si % tracklength
    idxmindist = findClosestS(si, sref)
    idxmindist2 = findSecondClosestS(si, sref, idxmindist)
    t = (si - sref[idxmindist]) / (sref[idxmindist2] - sref[idxmindist])
    x0 = (1 - t) * xref[idxmindist] + t * xref[idxmindist2]
    y0 = (1 - t) * yref[idxmindist] + t * yref[idxmindist2]
    psi0 = (1 - t) * psiref[idxmindist] + t * psiref[idxmindist2]

    x = x0 - ni * np.sin(psi0)
    y = y0 + ni * np.cos(psi0)
    psi = psi0 + alpha
    return np.vstack((x, y, psi, v))

def findClosestS(si, sref):
    si = np.atleast_1d(si)
    mindist = np.full(si.shape, np.inf)
    idxmindist = np.zeros(si.shape, dtype=int)
    for i in range(sref.size):
        di = np.abs(si - sref[i])
        idxmindist = np.where(di < mindist, i, idxmindist)
        mindist = np.where(di < mindist, di, mindist)
    idxmindist = np.where(idxmindist == sref.size, 1, idxmindist)
    idxmindist = np.where(idxmindist < 1, sref.size - 1, idxmindist)
    return idxmindist

def findSecondClosestS(si, sref, idxmindist):
    d1 = np.abs(si - sref[idxmindist - 1])
    d2 = np.abs(si - sref[(idxmindist + 1) % sref.size])
    idxmindist2 = np.where(d1 > d2, idxmindist + 1, idxmindist - 1)
    idxmindist2 = np.where(idxmindist2 == sref.size, 0, idxmindist2)
    idxmindist2 = np.where(idxmindist2 < 0, sref.size - 1, idxmindist2)
    return idxmindist2

def transformOrig2Proj(x, y, psi, v, filename='LMS_Track.txt'):
    [sref, xref, yref, psiref, _] = getTrack(filename=filename)
    idxmindist = findClosestPoint(x, y, xref, yref)
    idxmindist2 = findClosestNeighbour(x, y, xref, yref, idxmindist)
    t = findProjection(x, y, xref, yref, sref, idxmindist, idxmindist2)
    s0 = (1 - t) * sref[idxmindist] + t * sref[idxmindist2]
    x0 = (1 - t) * xref[idxmindist] + t * xref[idxmindist2]
    y0 = (1 - t) * yref[idxmindist] + t * yref[idxmindist2]
    psi0 = (1 - t) * psiref[idxmindist] + t * psiref[idxmindist2]

    s = s0
    n = np.cos(psi0) * (y - y0) - np.sin(psi0) * (x - x0)
    alpha = psi - psi0
    return np.vstack((s, n, alpha, v))

def findProjection(x, y, xref, yref, sref, idxmindist, idxmindist2):
    vabs = np.abs(sref[idxmindist] - sref[idxmindist2])
    vl = np.array([xref[idxmindist2] - xref[idxmindist], yref[idxmindist2] - yref[idxmindist]])
    u = np.array([x - xref[idxmindist], y - yref[idxmindist]])
    t = (vl[0] * u[0] + vl[1] * u[1]) / (vabs * vabs)
    return t

def findClosestPoint(x, y, xref, yref):
    mindist = np.inf
    idxmindist = 0
    for i in range(xref.size):
        dist = dist2D(x, xref[i], y, yref[i])
        if dist < mindist:
            mindist = dist
            idxmindist = i
    return idxmindist

def findClosestNeighbour(x, y, xref, yref, idxmindist):
    distBefore = dist2D(x, xref[idxmindist - 1], y, yref[idxmindist - 1])
    distAfter = dist2D(x, xref[(idxmindist + 1) % xref.size], y, yref[(idxmindist + 1) % xref.size])
    if distBefore < distAfter:
        idxmindist2 = idxmindist - 1
    else:
        idxmindist2 = idxmindist + 1
    if idxmindist2 < 0:
        idxmindist2 = xref.size - 1
    elif idxmindist2 >= xref.size:
        idxmindist2 = 0
    return idxmindist2

def dist2D(x1, x2, y1, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
