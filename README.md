Detection of LED head stage based on MTT files, used by the BBO lab at MPI for Neurobiology of Behavior

# Installation

## Windows

1. [Install Anaconda](https://docs.anaconda.com/anaconda/install/windows/)
2. Open Anaconda prompt via Start Menu
3. Create conda environment using `conda env create -f conda env create -f https://raw.githubusercontent.com/bbo-lab/multitrackpy/master/environment.yml`

# Usage

1. Switch to multitrackpy environment: `conda activate multitrackpy`
2. Run the program with `python -m multitrackpy -h`:
```
usage: __main__.py [-h] --mtt_file MTT_FILE --video_dir VIDEO_DIR
                   [--linedist_thres LINEDIST_THRES] [--corr_thres CORR_THRES]
                   [--led_thres LED_THRES] [--n_cpu N_CPU]
                   START_IDX END_IDX
__main__.py: error: the following arguments are required: START_IDX, END_IDX, --mtt_file, --video_dir
```
