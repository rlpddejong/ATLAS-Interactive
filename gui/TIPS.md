### Tips

Core mechanism: annotate objects at one or more frames and use propagation to complete the video.
Use permanent memory to store accurate segmentation (commit good frames to it) for best results.
The first frame to enter the memory bank is always committed to the permanent memory.
Reset memory if needed.

Controls:
- Use middle-click inside the image window to toggle between click-based segmentation and polygon-based segmentation.
- For click segmentation, use left-click for foreground annotation and right-click for background annotation.
- For polygon segmentation, use left-click to add points, and right-click to undo the last point.
- Use Ctrl+Z to undo the last click interaction or the last polygon point.
- Use number keys or the spinbox to change the object to be operated on. If it does not respond, most likely the correct number of objects was not specified during program startup.
- Use left/right arrows to move between frames, shift+arrow to move by 10 frames, and alt/option+arrow to move to the start/end.
- Use F/space and B to propagate forward and backward, respectively.
- Use T to toggle between visualization modes.
- Use C to commit a frame to permanent memory.
- Memory can be corrupted by bad segmentations. Make good use of "reset memory" and do not commit bad segmentations.
