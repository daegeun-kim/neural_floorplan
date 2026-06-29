Need update on current vectorization src:

axis alingment: 
- I see some points with very similar distance in one axis are not aligned. Make sure that all points (wall point, wall window end, wall door hinge, and wall door end) are aligned when the either vertical or horizontal distance is within 1000mm, regardless of the distance in the other axis. (eg: if the 2 points' x coordinate difference is 700mm, then regardless of y coordinate difference, x coordinate difference should be 0)

door generation: simplifying the process of door generation:
- instead of recognizing wall-door-hinge and wall-door-end point from base, make sure that those 2 points are 2 of the vertices of the bounding box of redpixels. (the hinge and end points are 2 of the 4 vertices of bbox). For choosing 2 points out of 4, the clue is purple pixels and black pixels around the red bbox.

ask me if any part of the instruction was not clear or need further judgment
Again, debug until all the must rules in C:\Users\kdgki\Desktop\MSCDP\Projects\neural_floorplan\specs\vectorization_must_rules.md are passed for samples in C:\Users\kdgki\Desktop\MSCDP\Projects\neural_floorplan\outputs\vectorization\v008\iteration5_run3