#!/usr/bin/env python3
"""Launch Vantage: python3 run_vantage.py [project.vantage | model.glb] [--sample]"""
import sys

from vantage.app import main

if __name__ == '__main__':
    sys.exit(main())
