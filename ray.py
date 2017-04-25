"""PPM file creator"""

import sys
import numpy as np
from numpy import linalg as LA


class Ray(object):
    """A ray class"""
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # TODO: Move this somewhere else...
    @staticmethod
    def normalize(vec):
        return vec/LA.norm(vec)

    def color(self):
        unit_direction = Ray.normalize(self.direction)
        t = 0.5*unit_direction[1] + 1.0
        return (1.0 - t)*np.ones(3) + t*np.array([0.5, 0.7, 1.0])


def main():
    """Main function"""
    nx = 200
    ny = 100
    sys.stdout.write("P3\n" + str(nx) + " " + str(ny) + "\n255\n")
    lower_left_corner = np.array([-2.0, -1.0, -1.0])
    horizontal = np.array([4.0, 0.0, 0.0])
    vertical = np.array([0.0, 2.0, 0.0])
    origin = np.zeros(3)
    for j in reversed(xrange(ny)):
        for i in range(nx):
            u = float(i) / float(nx)
            v = float(j) / float(ny)
            r = Ray(origin, lower_left_corner + u * horizontal + v*vertical)
            col = r.color()
            ir = int(255.99 * col[0])
            ig = int(255.99 * col[1])
            ib = int(255.99 * col[2])
            sys.stdout.write(str(ir) + " " + str(ig) + " " + str(ib) + "\n")


if __name__ == "__main__":
    main()
