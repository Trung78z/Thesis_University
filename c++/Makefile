# Makefile

TARGET = detection
ENGINE = models/best.engine
INPUT  = test_video.mp4
BUILD_DIR = build

# Default target
run: build
	@echo "Running $(TARGET)..."
	@$(BUILD_DIR)/$(TARGET) $(ENGINE) $(INPUT) || echo "❌ Program failed"

build:
	@echo "Building project with maximum optimization..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && \
	cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc)


debug:
	@echo "Building with debug symbols..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Debug .. && make -j$(nproc)

clean:
	@echo "Cleaning build directory..."
	@rm -rf $(BUILD_DIR)

rebuild: clean build
