from matplotlib import pyplot as plt
from NuRadioReco.detector import detector

if __name__ == "__main__":
    det = detector.Detector()
    fig, ax = plt.subplots(1, 1)
    for site in ["A", "B", "C", "D", "E", "F", "G", "X", "Y", "Z"]:
        pos = det.get_absolute_position_site(site)
        ax.plot(pos[0], pos[1], 'o')
        ax.annotate(site, [pos[0], pos[1]])

    ax.set_aspect("equal")
    ax.set_xlabel("easting [m]")
    ax.set_ylabel("northing [m]")
    fig.tight_layout()
    fig.savefig("map.png")
    plt.show()

