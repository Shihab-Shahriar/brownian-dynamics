# Compiler
CXX = nvcc 

# Compiler Flags
CXXFLAGS = -Xcompiler -Wall -O3 -std=c++17 -arch=sm_80

INCLUDES = -Iinclude

LIBS = 

# SFML Linker Flags
LDFLAGS = 
#-lsfml-graphics -lsfml-window -lsfml-system

# Executable name
TARGET1 = curand 
TARGET2 = openrand
TARGET3 = r123


# Source files
SOURCES = Curand.cu Openrand.cu r123.cu

# Object files
OBJECTS = $(SOURCES:.cu=.o)


all: $(TARGET1) $(TARGET2) $(TARGET3)
$(TARGET1): Curand.o
	$(CXX) $(CXXFLAGS) Curand.o -o $(TARGET1) $(LDFLAGS)

$(TARGET2): Openrand.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) Openrand.o -o $(TARGET2) $(LDFLAGS)

$(TARGET3): r123.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) r123.o -o $(TARGET3) $(LDFLAGS)

%.o: %.cu
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET1) $(TARGET2) $(TARGET3) out.txt a.out
