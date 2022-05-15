# Makefile for compiling the solver of Shu-Osher problem 
# compile/link all of the cpp files in SRC and create in BUILD

# Default target executed when no arguments are given to make.
default_target: clean all
	@echo
.PHONY: default_target

# General 
BUILD_DIR := ./build
SRC_DIR := ./src
MAIN := main.cpp

TARGET_EXEC := $(BUILD_DIR)/Shu-Osher

#=============================================================================
# Set variables for the build
# find all cpps in SRC_DIR and convert to SRC basename
CPPS := $(wildcard $(SRC_DIR)/*.cpp)
SRCS := $(CPPS:$(SRC_DIR)/%.cpp=%)
# or manually specify SRCS names
# SRCS := SparseMatrix 

# create OBJS path in BUILD
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)

# compiler
CC := g++
FLAGS := -std=c++11

# linking with external libraries
LIBS := armadillo
LFLAGS := $(LIBS:%=-l%)

#=============================================================================
# Targets
# Make the final executable file 
$(TARGET_EXEC): $(OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(FLAGS) $(OBJS) $(MAIN) -o $@ $(LFLAGS)

# Make required source objects
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CC) $(FLAGS) -c $< -o $@

all: $(OBJS) $(TARGET_EXEC)
.PHONY: all

clean:
	rm -rf $(BUILD_DIR)
.PHONY: clean
