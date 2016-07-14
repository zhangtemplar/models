#!/usr/bin/python
"""
This file creates dummy bounding boxes for cls_loc of the imagenet dataset.
It requires no xml files, but instead require the jpeg images available.
The output is a csv file, where each line likes:
nXXXXXXXX/nXXXXXXXX_YYYY.JPEG 0.0 0.0 1.0 1.0
i.e., we always assume the bounding box is the whole image
"""


import os
import sys


if __name__ == '__main__':
  if len(sys.argv) < 2 or len(sys.argv) > 3:
    print('Invalid usage\n'
          'usage: process_bounding_boxes.py <dir> [output-file]',
          file=sys.stderr)
    sys.exit(-1)
  with open(sys.argv[2], 'w') as fo:
    print 'start processing dataset from ' + sys.argv[1]
    total_processed = 0
    for directory in os.listdir(sys.argv[1]):
      print directory, ' being processed... ',
      num_processed = 0
      for filename in os.listdir(os.path.join(sys.argv[1], directory)):
        if not filename.endswith('.JPEG'):
          continue
        num_processed += 1
        image_filename = os.path.splitext(os.path.basename(filename))[0]
        fo.write(image_filename + '.JPEG,0.0,0.0,1.0,1.0\n')
      print num_processed, ' processed and done'
      total_procesed += num_processed
    print total_processed + ' images are processed and saved to ' + sys.argv[2]
