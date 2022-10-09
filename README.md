# image-editing-cli

Note: [] means optional and all values are taken as float unless mentioned otherwise

General Commands:
* load &lt;filename>.&lt;extension>
* save [&lt;filename>] [&lt;extension>]
* exit [save]
* show [&lt;filename>.&lt;extension>]
* undo
* redo

Image Manipulation Commands:
* greyscale
* invert
* solarize &lt;"<" or ">"> &lt;threshold value from 0 to 255>
* contrast &lt;value from -100 to 100>
* resize &lt;new number (integer) of rows> &lt;new number (integer) of columns>
* brightness &lt;value from -100 to 100>
* gamma correction &lt;gamma value>
* color pop &lt;color name in English> [invert]
* mean blur &lt;kernel size (integer)>
* gaussian blur &lt;kernel size (integer)> [&lt;sigma value, default sigma = 1>]
* bgr &lt;color name in English>

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
- [ ] saturation adjusting
