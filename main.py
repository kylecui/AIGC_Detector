"""Legacy entry point: `python main.py` delegates to the aigc-detector CLI."""
import sys

from aigc_detector.cli import main

if __name__ == "__main__":
    sys.exit(main())
