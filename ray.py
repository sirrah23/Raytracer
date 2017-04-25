"""PPM file creator"""

import sys

def main():
    """Main function"""
    nx = 200
    ny = 100
    sys.stdout.write("P3\n" + str(nx) + " " + str(ny) + "\n255\n")
    for j in reversed(xrange(ny)):
        for i in range(nx):
            r = float(i) / float(nx)
            g = float(j) / float(ny)
            b = 0.2
            ir = int(255.99*r)
            ig = int(255.99*g)
            ib = int(255.99*b)
            sys.stdout.write(str(ir) + " " + str(ig) + " " + str(ib) + "\n")


if __name__ == "__main__":
    main()
