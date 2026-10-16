"""Cache frozen DINOv3 patch tokens for the human videos or one folder of simulated episodes."""
import argparse

from src.config import DEMO_DIR, PLAY_DIR, ROBOTIQ_DIR
from src.features.dino import extract_videos
from src.features.sim import extract_episodes

SIM_SOURCES = {"panda_demos": DEMO_DIR, "panda_play": PLAY_DIR, "robotiq_play": ROBOTIQ_DIR}


def main():
    """Extract features for the chosen source."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["human_videos", *SIM_SOURCES])
    source = parser.parse_args().source
    extract_videos() if source == "human_videos" else extract_episodes(SIM_SOURCES[source])


if __name__ == "__main__":
    main()
