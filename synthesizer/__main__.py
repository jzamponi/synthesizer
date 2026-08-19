#!/usr/bin/env python3
"""
Pipeline script to calculate synthetic ALMA/JVLA images from an SPH model,
using RADMC3D for the raytracing and CASA for the synthetic observation.
"""

from synthesizer.parser import synthesizer

if __name__ == "__main__":
    synthesizer()
