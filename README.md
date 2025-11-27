# RISC-V simulator

## Building the Project

The code base is written in C++17, to build the project use cmake. (You might want to use 
ninja for faster builds.)

## Usage

To run the simulator, use the following command:

```
./vm --start-vm
```

See [Commands](COMMANDS.md) for a list of commands.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.


## References
- [RISC-V Specifications](https://riscv.org/specifications/)
- [Five EmbedDev ISA manual](https://five-embeddev.com/riscv-isa-manual/)

## building for mac.os
- mkdir build
- cd build
- cmake -DCMAKE_C_COMPILER=/opt/homebrew/bin/gcc-14 \
      -DCMAKE_CXX_COMPILER=/opt/homebrew/bin/g++-14 \
      ..
- make -j4

## building for linux
- mkdir build
- cd build
- cmake ..
- make -j4
