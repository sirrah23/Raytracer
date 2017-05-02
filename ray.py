"""PPM file creator"""

import sys
import numpy as np
import math
from numpy import linalg as LA
from random import random

class HitRecord(object):
    def __init__(self, t, p, normal, material):
        self.t = t
        self.p = p
        self.normal = normal
        self.material = material


class Hitable(object):
    def hit(self, ray, t_min, t_max, rec):
        raise NotImplementedError()


class Sphere(Hitable):

    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

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
                return (True, HitRecord(temp, p, normal, self.material))
            else:
                temp = (-b + (b*b - a*c)**0.5)/a
                if t_min < temp < t_max:
                    p = ray.point_at_parameter(temp)
                    normal = (p - self.center) / self.radius
                    return (True, HitRecord(temp, p, normal, self.material))
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
    def __init__(self, vfov, aspect, lookfrom, lookat, vup):
        theta = vfov*math.pi/180
        half_height = math.tan(theta/2)
        half_width = aspect * half_height
        self.origin = lookfrom
        w = Ray.normalize(lookfrom - lookat)
        u = Ray.normalize(np.cross(vup, w))
        v = np.cross(w, u)
        self.lower_left_corner = self.origin - half_width*u - half_height*v - w
        self.horizontal = 2*half_width*u
        self.vertical = 2*half_height*v

    def generate_ray(self, u, v):
        return Ray(self.origin, self.lower_left_corner + u*self.horizontal + v*self.vertical - self.origin)

class Material(object):
    def scatter(ray, hit_record, attenuation, scattered):
        raise NotImplementedError()


class Lambertian(Material):

    def __init__(self, albedo):
        self.albedo = albedo

    def scatter(self, ray, hit_record):
        target = hit_record.p + hit_record.normal + random_in_unit_sphere()
        scattered = Ray(hit_record.p, target-hit_record.p)
        return (True, scattered, self.albedo)


def reflect(vec, vec_normal):
    return vec - 2 * np.dot(vec, vec_normal) * vec_normal

def refract(vec, vec_normal, ni_over_nt):
    uv = Ray.normalize(vec)
    dt = np.dot(uv, vec_normal)
    discriminant = 1.0 - ni_over_nt**2*(1-dt**2)
    if discriminant > 0:
        refracted = ni_over_nt*(uv-vec_normal*dt)-vec_normal*discriminant**0.5
        return (True, refracted)
    else:
        return (False, None)

def schlick(cos, ref_idx):
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0**2
    return r0 + (1-r0)*(1-cos)**5

class Metal(Material):

    def __init__(self, albedo, fuzz):
        self.albedo = albedo
        self.fuzz = fuzz

    def scatter(self, ray, hit_record):
        reflected = reflect(Ray.normalize(ray.direction), hit_record.normal)
        scattered = Ray(hit_record.p, reflected + self.fuzz * random_in_unit_sphere())
        if np.dot(scattered.direction, hit_record.normal) > 0:
            return (True, scattered, self.albedo)
        else:
            return (False, None, self.albedo)


class Dielectric(Material):

    def __init__(self, ref_idx):
        self.ref_idx = ref_idx

    def scatter(self, ray, hit_record):
        reflected = reflect(ray.direction, hit_record.normal)
        attenuation = np.ones(3)
        if np.dot(ray.direction, hit_record.normal) > 0:
            outward_normal = -hit_record.normal
            ni_over_nt = self.ref_idx
            cos = self.ref_idx * np.dot(ray.direction, hit_record.normal) / LA.norm(ray.direction)
        else:
            outward_normal = hit_record.normal
            ni_over_nt = 1.0 / self.ref_idx
            cos = -np.dot(ray.direction, hit_record.normal) / LA.norm(ray.direction)
        ref_status, refracted = refract(ray.direction, outward_normal, ni_over_nt)
        if ref_status:
            reflect_prob = schlick(cos, self.ref_idx)
        else:
            scattered = Ray(hit_record.p, reflected)
            reflect_prob = 1.0
        if random() < reflect_prob:
            scattered = Ray(hit_record.p, reflected)
        else:
            scattered = Ray(hit_record.p, refracted)
        return (ref_status, scattered, attenuation)


def color(ray, world, depth):
    hit_status, rec = hit_list(world, ray, 0.001, float("inf"))
    if hit_status:
        should_scatter, scattered_ray, attenuation = rec.material.scatter(ray, rec)
        if depth < 50 and should_scatter:
            return attenuation * color(scattered_ray, world, depth + 1)
        else:
            return np.zeros(3)
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
    world.append(Sphere(np.array([0, 0, -1]), 0.5, Lambertian(np.array([0.8, 0.3, 0.3]))))
    world.append(Sphere(np.array([0, -100.5, -1]), 100, Lambertian(np.array([0.8, 0.8, 0.0]))))
    world.append(Sphere(np.array([1, 0, -1]), 0.5, Metal(np.array([0.8, 0.6, 0.2]), 0.3)))
    world.append(Sphere(np.array([-1, 0, -1]), 0.5, Dielectric(1.5)))
    cam = Camera(90, float(nx)/float(ny), np.array([-2,2,1]),np.array([0,0,-1]),np.array([0,1,0]))
    for j in reversed(xrange(ny)):
        for i in range(nx):
            col = np.zeros(3)
            for _ in range(ns):
                u = float(i + random()) / float(nx)
                v = float(j + random()) / float(ny)
                r = cam.generate_ray(u, v)
                #p = r.point_at_parameter(2.0)
                col += color(r, world, 0)
            col /= float(ns)
            col = col**0.5
            ir = int(255.99 * col[0])
            ig = int(255.99 * col[1])
            ib = int(255.99 * col[2])
            sys.stdout.write(str(ir) + " " + str(ig) + " " + str(ib) + "\n")


if __name__ == "__main__":
    main()
