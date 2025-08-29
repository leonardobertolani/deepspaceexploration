# aca25-deepspace-fpga
ACA 2025 project made by Samuele Tondelli and Leonardo Bertolani.

## Description
The goal of this project was making an accelator on the fpga of the Kria KD240 to accelerate the resolution of SOCP problems on embedded platforms, which is required for the emerging problem of autonomous deep space exploration. We focused on Ecos, an open source library designed to solve said problems, and decided to accelerate 5 specific functions:
- **spmv**, sparse matrix vector multiplication
- **spmtvm**, spmv with the matrix transposed
- **lsolve**, solves the L part of an LDLt system
- **dsolve**, solves the D part of an LDLt system
- **ltsolve**, solves the Lt part of an LDLt system

We developed the project using Vitis HLS, and creatd a monolithic accelerator called **xlr8_vec**.

## Structure of the repository
- **bitstream**, contains the necessary files to upload the bitstream
- **ecos-x**, contains the modified version of ecos using our accelerator, with the bindings found in ```external/xlr8```
- **images**, contains the images for readme
- **logs**, contains the source to measure the power consumption
- **power_hw**, contains the sources used to generate the power measurements
- **xlr8**, contains the HLS sources, with files finishing in ```_tb``` being the testbench files
- **xrt_text**, contains the benchmarks for the single functions

## Usage
For development, we used Vitis 2024.2, building the IP with the following process:
- Create a new component, with target platform ```xck24-ubva530-2LV-c``` and target clock ```7ns``` with uncertainty ```1ns```
- Add to the source all non testbench files in every folder finishing with _vec, and optionally add the ```xlr8_vec_tb.cpp``` as testbench
- Run the Synthesis and the Package step in Vitis to create the IP

To actually generate the bitstream, we used Vivado 2024.2 with the following process:
- Create a new project with target platform Kria KD240
- Import the generated IP and create a new block design, enabling all the slave AXI HP and setting the PL fabric clocks PL0 and PL1 to 250 MHz, you should create a block design similar to this:

![Block design](./images/block_design.png)

- Then create an HDL wrapper and run the Synthesis and Implementation, as strategies we used respectively ```Flow_PerfOptimized_high``` and ```Performance_ExplorePostRoutePhysOpt```
- Then generate the bitstream and export the hardware to get the necessary files, which can also be found in this repository under ```bitstream```

**Note**: the generated bitstream has a negative WNS, but from testing we didn't find and sign of instability.

Once the necessary files are generated, upload them to the Kria using the Jupyter Notebook of the PYNQ enviroment like so:

```python
from pynq import Overlay
import pynq
import numpy as np
import time
ov = Overlay("./deep_space_vec_250_wrapper.bit")
```

## Benchmarks
All of the benchmarks can be built with their respective Makefile by running ```make```. To test ecos-x, we built it with ```make all``` and run the example ```runecos```. Only ```logs/log_power.c``` doesn't have a Makefile, but can be built with
```
gcc -o log_power log_power.c -O3
```

