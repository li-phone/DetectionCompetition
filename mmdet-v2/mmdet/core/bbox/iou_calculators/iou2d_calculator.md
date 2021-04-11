
Yes, we proved it as follows: 

**Prove:**

Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou', there are some **new generated variable** when calculating IOU using bbox_overlaps function:

1) is_aligned is False


    area1: M x 1    
    area2: N x 1
    lt: M x N x 2
    rb: M x N x 2
    wh: M x N x 2
    overlap: M x N x 1
    union: M x N x 1
    ious: M x N x 1
    
Total memory:

    S = (9 x N x M + N + M) * 4 Byte

When using FP16, we can reduce:

    R = (9 x N x M + N + M) * 4 / 2 Byte,

R large than (N + M) * 4 * 2 **is always true** when N and M >= 1.

    Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
               N + 1 < 3 * N, when N or M is 1.

Given M = 40 (ground truth), N = 400000 (three anchor boxes in per grid, FPN, R-CNNs),

    R = 275 MB (one times)

A special case (**dense detection**), M = 512 (ground truth),
    
    R = 3516 MB = 3.43 GB

When the batch size is B, reduce:
    
    B x R

Therefore, **CUDA memory runs out frequently**.

Experiments on GeForce RTX 2080Ti (11019 MiB):

|   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
|:----:|:----:|:----:|:----:|:----:|:----:|
|   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
|   FP16   |   512 | 400000 |   4504 MiB | **3516 MiB** | **3516 MiB** |
|   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
|   FP16   |   40 | 400000 |   1264 MiB |   **276MiB**   | **275 MiB** |

2) is_aligned is True


    area1: N x 1
    area2: N x 1
    lt: N x 2
    rb: N x 2
    wh: N x 2
    overlap: N x 1
    union: N x 1
    ious: N x 1

Total memory:

    S = 11 x N * 4 Byte

When using FP16, we can reduce:

    R = 11 x N * 4 / 2 Byte

So do the 'giou' (**large than 'iou'**).

Time-wise, **FP16 is generally faster than FP32.**

When gpu_assign_thr is not -1, **it takes more time on cpu but not reduce memory.**

Therefore, **we can reduce half the memory and keep the speed.**




