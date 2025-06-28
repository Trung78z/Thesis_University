# Variables
TARGET           := detection
ENGINE           := models/model-28-6.engine
VIDEO_INPUT      := videos/test_video.mp4
BUILD_DIR        := build
BIN_DIR          := bin
CMAKE_BUILD_TYPE ?= Release

# Default target
.PHONY: all run build clean rebuild debug

all: run

run: build
	@echo "🚀 Running $(TARGET) with video input..."
	@$(BIN_DIR)/$(TARGET) --engine $(ENGINE) --video $(VIDEO_INPUT) || echo "❌ Program failed"

build:
	@echo "🔧 Building project..."
	@cmake -B $(BUILD_DIR) -S . -DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE)
	@cmake --build $(BUILD_DIR) --target $(TARGET) -- -j$(shell nproc)

rebuild: clean build

clean:
	@echo "🧹 Cleaning build files..."
	@rm -rf $(BUILD_DIR) $(BIN_DIR)

debug:
	@$(BIN_DIR)/$(TARGET) --help
