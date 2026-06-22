# main.py — single entry point for the SCARABot visualizers
import sys
import visualize2D
import visualize3D


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "2d"
    if mode == "3d":
        visualize3D.launch()
    else:
        visualize2D.launch()
    


if __name__ == "__main__":
    main()