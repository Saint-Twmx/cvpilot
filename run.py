from segmentation.nnUNet_inference import main as infer_main
from measure.main import main
import os

if __name__ == "__main__":
    infer_main(os.path.dirname(os.path.abspath(__file__)), "Demo.nii.gz")
    main()