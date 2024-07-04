### Requirements:
1. Install `requirements.txt`
2. `cd` to the clone's main directory

### How to execute:

**usage:**

`python style_transfer.py [-h] [--num_inference_steps NUM_INFERENCE_STEPS [NUM_INFERENCE_STEPS ...]] [--alpha ALPHA [ALPHA ...]] [--eta ETA] [--dataset {1,2,3}] [--test_idx TEST_IDX]`

**options:**

  `-h`, `--help`            show this help message and exit
  
  `--num_inference_steps NUM_INFERENCE_STEPS [NUM_INFERENCE_STEPS ...]`
                        Number of inference steps to sweep over
                        
  `--alpha ALPHA [ALPHA ...]`
                        Alpha values to sweep over
                        
  `--eta ETA`             Eta value
  
  `--dataset {1,2,3}`     1: horse2zebra, 2: open2close, 3: black2blue
  
  `--test_idx TEST_IDX`   Index of test image out of the dataset

  ### Result:
  1. Output figure containing sweeped parameters will be created in the same directory, with the prefix `sweep_result--` and suffix `.png`
