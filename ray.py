"""PPM file creator"""

import sys
import numpy as np
from numpy import linalg as LA
from random import random

class HitRecord(object):
    def __init__(self, t, p, normal):
        self.t = t
        self.p = p
        self.normal = normal


class Hitable(object):
    def hit(self, ray, t_min, t_max, rec):
        raise NotImplementedError()


class Sphere(Hitable):

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def hit(self, ray, t_min, t_max):
        """Determine if a ray hits a sphere with center and radius"""
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius**2
        discriminant = b**2 - 4*a*c
        if discriminant > 0:
            temp = (-b - (b*b - a*c)**0.5)/a
            if t_min < temp < t_max:
                p = ray.point_at_parameter(temp)
                normal = (p - self.center) / self.radius
                return (True, HitRecord(temp, p, normal))
            else:
                temp = (-b + (b*b - a*c)**0.5)/a
                if t_min < temp < t_max:
                    p = ray.point_at_parameter(temp)
                    normal = (p - self.center) / self.radius
                    return (True, HitRecord(temp, p, normal))
        return (False, None)


def hit_list(hittable_list, ray, t_min, t_max):
    closest_rec = None
    hit_anything = False
    closest_so_far = t_max
    for hittable in hittable_list:
        hit_status, rec = hittable.hit(ray, t_min, closest_so_far)
        if hit_status:
            hit_anything = True
            closest_so_far = rec.t
            closest_rec = rec
    return hit_anything, closest_rec

class Ray(object):
    """A ray class"""
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def point_at_parameter(self, t):
        return self.origin + self.direction * t

    # TODO: Move this somewhere else...
    @staticmethod
    def normalize(vec):
        return vec/LA.norm(vec)

    @staticmethod
    def squared_dist(vec):
        return LA.norm(vec)**2

class Camera(object):
    def __init__(self, lower_left_corner, horizontal, vertical, origin):
        self.lower_left_corner = lower_left_corner
        self.horizontal = horizontal
        self.vertical = vertical
        self.origin = origin

    def generate_ray(self, u, v):
        return Ray(self.origin, self.lower_left_corner + u*self.horizontal + v*self.vertical - self.origin)

def color(ray, world):
    hit_status, rec = hit_list(world, ray, 0.001, float("inf"))
    if hit_status:
        target = rec.p + rec.normal + random_in_unit_sphere() # matte
        return 0.5*color(Ray(rec.p, target-rec.p), world)
    else:
        unit_dir = Ray.normalize(ray.direction)
        t = 0.5*(unit_dir[1] + 1)
        return (1.0-t)*np.ones(3) + t*np.array([0.5, 0.7, 1.0])

def random_in_unit_sphere():
    rand_pt = 2.0 * np.array([random(), random(), random()]) - np.ones(3)
    while(Ray.squared_dist(rand_pt) >= 1):
        rand_pt = 2.0 * np.array([random(), random(), random()]) - np.ones(3)
    return rand_pt

def main():
    """Main function"""
    nx = 200
    ny = 100
    ns = 100
    sys.stdout.write("P3\n" + str(nx) + " " + str(ny) + "\n255\n")
    world = []
    world.append(Sphere(np.array([0, 0, -1]), 0.5))
    world.append(Sphere(np.array([0, -100.5, -1]), 100))
    lower_left_corner = np.array([-2.0, -1.0, -1.0])
    horizontal = np.array([4.0, 0.0, 0.0])
    vertical = np.array([0.0, 2.0, 0.0])
    origin = np.zeros(3)
    cam = Camera(lower_left_corner, horizontal, vertical, origin)
    for j in reversed(xrange(ny)):
        for i in range(nx):
            col = np.zeros(3)
            for _ in range(ns):
                u = float(i + random()) / float(nx)
                v = float(j + random()) / float(ny)
                r = cam.generate_ray(u, v)
                #p = r.point_at_parameter(2.0)
                col += color(r, world)
            col /= float(ns)
            col = col**0.5
            ir = int(255.99 * col[0])
            ig = int(255.99 * col[1])
            ib = int(255.99 * col[2])
            sys.stdout.write(str(ir) + " " + str(ig) + " " + str(ib) + "\n")


if __name__ == "__main__":
    main()
