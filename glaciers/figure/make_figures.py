#!/usr/bin/env python3
"""Draw the figures of Sect. 5 of the paper into this folder:
  fig_gmd_glaciers.py           -> fig_04_glaciers_grounding_lines   (Fig. 4)
  fig_gmd_performance.py        -> fig_05_accuracy_and_cost          (Fig. 5; also rewrites data/summary.csv)
  fig_gmd_speed_difference.py   -> fig_06_speed_difference           (Fig. 6)
  fig_gmd_geometry_response.py  -> fig_08_geometry_response          (Fig. 8)
  fig_gmd_profiles.py           -> fig_C1_flowline_profiles          (Fig. C1; also the preprint layout, _wide)
Run from anywhere: python make_figures.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_gmd_geometry_response  # noqa: E402
import fig_gmd_glaciers  # noqa: E402
import fig_gmd_performance  # noqa: E402
import fig_gmd_profiles  # noqa: E402
import fig_gmd_speed_difference  # noqa: E402

if __name__ == "__main__":
    sys.argv = sys.argv[:1]
    for module in (fig_gmd_glaciers, fig_gmd_performance, fig_gmd_speed_difference, fig_gmd_geometry_response):
        module.main()
    fig_gmd_profiles.main(wide=False)
    fig_gmd_profiles.main(wide=True)
