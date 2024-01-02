import unittest
import os
from aos_wrapper import generate_integral

class TestAOS(unittest.TestCase):
    def test_aos_integrator(self):
        batch_index = 1
        sample_index = 320
        required_file = f"../data/train/{batch_index}_{sample_index}_pose_0_thermal.png"

        no_train_error_msg = f"Please ensure you have set up training data using the preprocess.py and that the file {required_file} exists."
        if not os.path.isdir("../data/train"):
            raise FileNotFoundError(no_train_error_msg)
        if not os.path.isfile(required_file):
            raise FileNotFoundError(no_train_error_msg)

        out_image = f"../data/train/{batch_index}_0-aos_thermal-0.png"
        if os.path.isfile(out_image):
            print("Deleting existing file.")
            os.remove(out_image)
        generate_integral(0, 0, focal_planes=[], _dir_adjust="..")

        if not os.path.isfile(out_image):
            raise FileNotFoundError("Couldn't find AOS output file, something went wrong.")


