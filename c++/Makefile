run:
	./build/detection models/car_d.engine test_video.mp4
clean:
	rm -rf build
build:
	mkdir -p build
	cd build && cmake .. && make -j$(nproc)