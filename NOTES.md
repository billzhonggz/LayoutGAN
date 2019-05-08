# Notes

Notes on testing the network.

### May 8 2019 11:00 CST

Commit `bd1f56e5da4471d836b3da6d18e7e86a3a198bc6`

- Disable batch norm layers and max pooling layers.
- Bypass the errors before.
- A new error in loss passing:
    - `Target size (torch.Size([15])) must be the same as input size (torch.Size([15, 128]))`
    - On `main.py`, `real_loss` and `fake_loss` methods.

### May 1 2019 22:30 CST

Commit `f7990a096dd61240d47e7ae382c9e4c920fc35a7`

**Some progress.**

- Figure out of memory issue: it IS because my VRAM is too small.
- FIXME
    - The fake images has the element num equals to batch size, consider to extract "element_num" parameter to other places.

### May 1 2019 17:00 CST

Commit `f3fe546f03ec3bb041d5af02d675a490b61e644d`

**Running failed**

- The program stopped due to out of memory issue.
- Guesses
    - The program consumes VRAM slowly may due to low efficient on the data set handling function.
    - The program run out of the VRAM and system RAM may due to the checkpoints were not saved automatically.
- Something to try
    - Parallelize the data set handling function.
    - Explore how PyTorch can save something automatically.