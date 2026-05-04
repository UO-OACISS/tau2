include ../../include/Makefile

PTISRC_DIR := $(TAUROOT)/utils/pti_elf_parser/src
PTIINC_DIR := $(TAUROOT)/utils/pti_elf_parser/include

CXXFLAGS := -I$(PTIINC_DIR) -I$(TAUROOT)/utils/pti_elf_parser/include -O3 -fPIC -L/$(TAU_L0_LIGA_DIR)  -liga64 -Wl,rpath,$(TAU_L0_LIGA_DIR)

TARGET := Pti_elf_parser.o

OBJS := \
        elf_parser.o \
        section_debug_line.o \
        section_debug_info.o \
        section_debug_abbrev.o \
        dwarf_state_machine.o

all: $(TARGET)


elf_parser.o: $(PTISRC_DIR)/elf_parser.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

section_debug_line.o: $(PTISRC_DIR)/section_debug_line.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

section_debug_info.o: $(PTISRC_DIR)/section_debug_info.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

section_debug_abbrev.o: $(PTISRC_DIR)/section_debug_abbrev.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

dwarf_state_machine.o: $(PTISRC_DIR)/dwarf_state_machine.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(LD) -r -o $@ $^

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean