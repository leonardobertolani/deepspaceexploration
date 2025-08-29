#!/bin/bash

clear
cat << "EOF"

        +                               .                       +
                              *                  *                 .
      .         .                  .                  .                  +         *
                      .
██████╗  ███████╗ ███████╗ ██████╗      ███████╗ ██████╗   █████╗  ██████╗  ███████╗
██╔══██╗ ██╔════╝ ██╔════╝ ██╔══██╗    ██╔════╝ ██╔══██╗ ██╔══██╗ ██╔════╝ ██╔═════╝
██║  ██║ █████╗   █████╗   ██████╔╝    ███████╗ ██████╔╝ ███████║ ██║      █████╗
██║  ██║ ██╔══╝   ██╔══╝   ██╔══╝      ╚════██║ ██╔══╝   ██╔══██║ ██║      ██╔══╝
██████╔╝ ███████╗ ███████╗ ██║         ███████║ ██║      ██║  ██║ ████████ ███████╗
╚═════╝  ╚══════╝ ╚══════╝ ╚═╝         ╚══════╝ ╚═╝      ╚═╝  ╚═╝ ╚══════╝ ╚══════╝


                      E X P L O R A T I O N
    .                                                 +                          *
                  .              +                             .         
        *                 .           .         *       
 .                          .                        .                   *    .
            +                             .                        +

EOF

cd xrt_test
echo "Building xrt_test"
make clean
make
echo "Building xrt_test completed"
cd ..

cd power_hw
echo "Building power_hw"
make clean
make
echo "Building power_hw completed"
cd ..

cd logs
echo "Building logs"
gcc -c log_power.c -o log_power -Wno-format
echo "Building logs completed"
cd ..


echo "Loading bitstream"
source '/usr/local/share/pynq-venv/bin/activate'
source '/home/ubuntu/pynq/Kria-PYNQ/pynq/sdbuild/packages/xrt/xrt_setup.sh'
/usr/local/share/pynq-venv/bin/python3 -c 'from pynq import Overlay; Overlay("./bitstream/deep_space_vec_250_wrapper.bit");'
echo "Bitstream ready"

sudo ./xrt_test/build/time_bench
