hdl
===

Hierarchical Directed Learners (HDL)

---

Quick Start
===========

### Install:

`python setup.py develop --user`

### Get some data:

`wget http://redwood.berkeley.edu/cadieu/data/vid075-chunks.tar.gz .`
`tar -zxf vid075-chunks.tar.gz /share/users/USERNAME/data/vid075-chunks`

### Run the code (from within hdl/scripts):

`python learn_sparsemodel.py`

or with gpu:

`THEANO_FLAGS=mode=FAST_RUN,device=gpu${GPU_NUM},floatX=float32 python learn_sparsemodel.py`

### View the results:

`/share/users/USERNAME/output/hdl/state/.../*.png`

### Ariel:

Split images:

`ffmpeg -i ../wak/hdr_wak_003.avi -r 1000 -f image2 image-%07d.png`
