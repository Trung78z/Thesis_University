run:
	./build/detection car_detect.engine test_video.mp4
clean:
	rm -rf build
build:
	mkdir -p build
	cd build && cmake .. && make -j$(nproc)