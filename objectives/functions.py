"""Benchmark functions: scalar f(x)->float and vectorised f_vec(X)->ndarray."""
from __future__ import annotations
import math, numpy as np

def sphere(x): return float(np.sum(x**2))
def sphere_vec(X): return np.sum(X**2, axis=1)

def rosenbrock(x): return float(np.sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2))
def rosenbrock_vec(X): return np.sum(100*(X[:,1:]-X[:,:-1]**2)**2 + (1-X[:,:-1])**2, axis=1)

def rastrigin(x):
    return float(10*len(x) + np.sum(x**2 - 10*np.cos(2*math.pi*x)))
def rastrigin_vec(X):
    return 10*X.shape[1] + np.sum(X**2 - 10*np.cos(2*np.pi*X), axis=1)

def ackley(x):
    d = len(x)
    return float(-20*math.exp(-0.2*math.sqrt(np.sum(x**2)/d)) - math.exp(np.sum(np.cos(2*math.pi*x))/d) + 20 + math.e)
def ackley_vec(X):
    d = X.shape[1]
    return -20*np.exp(-0.2*np.sqrt(np.sum(X**2,axis=1)/d)) - np.exp(np.sum(np.cos(2*np.pi*X),axis=1)/d) + 20 + np.e

# Registry mapping scalar -> vectorised
VEC_REGISTRY = {sphere: sphere_vec, rosenbrock: rosenbrock_vec, rastrigin: rastrigin_vec, ackley: ackley_vec}
