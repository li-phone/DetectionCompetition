from batch_train import main as bat_train_main
from batch_infer import main as bat_infer_main
from orange2_parallel_inference import main as infer_main
from orange2_parallel_inference2 import main as infer_main2
from orange2_parallel_inference3 import main as infer_main3

bat_train_main()
infer_main3("../configs/orange2/cas_r101-best_base-800x800_1000x1000_ovlap_0.5-resize_0.5_1.0-v2.py")
infer_main2("../configs/orange2/cas_r101-best_base-800x800_1000x1000_ovlap_0.5-resize_0.5_1.0.py")
infer_main("../configs/orange2/cas_r101-best_base-800x800_1000x1000_ovlap_0.5-resize_0.5_1.0_1.5.py")
