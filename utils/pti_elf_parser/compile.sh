#!/bin/bash

icpx -I${PWD}/include \
     -O3 -fPIC \
     -o elf_parser.o \
     -c src/elf_parser.cpp

icpx -I${PWD}/include \
     -O3 -fPIC \
     -o section_debug_line.o \
     -c src/section_debug_line.cpp

icpx -I${PWD}/include \
     -O3 -fPIC \
     -o section_debug_info.o \
     -c src/section_debug_info.cpp

icpx -I${PWD}/include \
     -O3 -fPIC \
     -o section_debug_abbrev.o \
     -c src/section_debug_abbrev.cpp

icpx -I${PWD}/include \
     -O3 -fPIC \
     -o dwarf_state_machine.o \
     -c src/dwarf_state_machine.cpp

icpx -shared -o libdebug_info_parser.so \
     elf_parser.o \
     section_debug_line.o \
     section_debug_info.o \
     section_debug_abbrev.o \
     dwarf_state_machine.o
