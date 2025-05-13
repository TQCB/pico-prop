# Compiler and flags
CC = gcc
# CFLAGS:
#   -Wall: Enable all warnings
#   -g: Add debug symbols
#   -O2: Optimization level 2 (good balance)
#   -fPIC: Position Independent Code (required for shared libraries)
CFLAGS = -Wall -g -O2 -fPIC
# INCLUDES: Path to your header files
INCLUDES = -Iinclude
# LDFLAGS: Linker flags
#   -shared: Create a shared library
LDFLAGS = -shared
# LIBS: Libraries to link against
#   -lm: Math library (for sin, cos, exp, pow, etc.)
LIBS = -lm

# Directories
SRCDIR = src
BUILDDIR_OUT = ../build

# Source files
SOURCES = $(SRCDIR)/ad_engine.c

# --- Target Naming ---
# Default to Linux behavior
TARGET_NAME_BASE = libad_engine
SHARED_LIB_EXT = .so

# Final target library path
TARGET = $(BUILDDIR_OUT)/$(TARGET_NAME_BASE)$(SHARED_LIB_EXT)

# --- Rules ---

# Phony targets are not files
.PHONY: all clean

# Default target: build the shared library
all: $(TARGET)

# Rule to build the shared library
$(TARGET): $(SOURCES)
	@mkdir -p $(BUILDDIR_OUT)
	@echo "Compiling $(SOURCES) into $(TARGET)..."
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $^ $(LIBS)
	@echo "Successfully built $(TARGET)"

# Rule to clean build artifacts
clean:
	@echo "Cleaning build directory..."
	@rm -rf $(BUILDDIR_OUT)/$(TARGET_NAME_BASE)$(SHARED_LIB_EXT)
	@rm -f $(SRCDIR)/*.o
	@echo "Clean complete."

# test rule
# test: all
#    python3 your_test_script.py