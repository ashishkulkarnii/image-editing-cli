# image-editing-cli

Note: [] means optional
      all values are taken as float unless mentioned otherwise
General Commands:
* load <filename>.<extension>
* save [<filename>] [<extension>]
* exit [save]
* show [<filename>.<extension>]
* undo
* redo
Image Manipulation Commands:
* greyscale
* invert
* solarize <"<" or ">"> <threshold value from 0 to 255>
* contrast <value from -100 to 100>
* resize <new number (integer) of rows> <new number (integer) of columns>
* brightness <value from -100 to 100>
* gamma correction <gamma value>
* color pop <color name in English> [invert]
* mean blur <kernel size (integer)>
* gaussian blur <kernel size (integer)> [<sigma value, default sigma = 1>]
* bgr <color name in English>

## Functionalities:
- [x] greyscale conversion
- [x] inverting
- [x] solarizing
- [x] contrast sdjusting
- [x] resizing
- [x] brightness adjusting
- [ ] temperature adjusting
- [x] gamma correction
- [x] color pop
- [x] mean blur
- [x] gaussian blur
- [x] color name in English to BGR
- [ ] portrait mode
- [ ] masking based on edge detection
