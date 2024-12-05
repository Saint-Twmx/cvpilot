from segmentation.nnUNet_inference import main as infer_main
from measure.main import main


if __name__ == "__main__":
    infer_main(r"./cvpilot", "Demo.nii.gz")
    main()