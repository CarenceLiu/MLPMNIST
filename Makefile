# Compiler and Compiler Flags, from GPT-4
CXX = g++
# CXXFLAGS = -std=c++17 -Wall
CXXFLAGS = -std=c++17 -O3

# To add any dependencies from the source folder
DEPS = source/dataLoader.cpp source/logger.cpp source/mlp.cpp
# Header files directory
INCLUDES = -Isource

# Specify the target binaries (from the test directories)
TARGETS = test_1/test_1 \
        #   test_2_1/test_2_1 \
        #   test_2_2/test_2_2 \
        #   test_2_3/test_2_3 \
        #   test_2_4/test_2_4 \
        #   test_2_5/test_2_5 \
        #   test_2_6/test_2_6

# Default target
all: $(TARGETS)

$(TARGETS): % : %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(DEPS) -o $@

# Clean
clean:
	rm -f $(TARGETS)
