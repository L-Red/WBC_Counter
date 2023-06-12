'''
This file contains the main function for stitching images captured from the microscope. It is important
that the images need to be captured in a specific order such that the stitching algorithm can work. Images need
to be adjacent to each other in order for the stitching algorithm to work. For example, if the images are captured
in a 3x3 grid, the images need to be captured in the following order:

    1 2 3
    4 5 6
    7 8 9

The stitching algorithm will not work if the images are captured in the following order:

    1 2 3
    4 5 6
    9 8 7

This is because the stitching algorithm will not be able to find overlapping features between images 6 and 9.
Maybe in the future, we can implement a more robust stitching algorithm that can handle images that are not
adjacent to each other. Right now, we are dependent on the stitching library.
'''

import stitching
stitcher = stitching.Stitcher()


class MyStitcher(stitching.Stitcher):

    def __int__(self):
        super(MyStitcher, self).__init__()

    def __call__(self, *args, **kwargs):
        return self.stitch(*args, **kwargs)