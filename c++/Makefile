build:
	cd build 
		&& \
		$(MAKE) clean
		&& \
		cmake ..
		&& \
		$(MAKE) -j 4
clean:
	cd build && \
		$(MAKE) clean
	rm -rf build
	rm -rf install
	rm -rf lib
	rm -rf bin
	rm -rf include
	rm -rf CMakeCache.txt
	rm -rf CMakeFiles
	rm -rf cmake_install.cmake
	rm -rf Makefile
	rm -rf CMakeError.log
	rm -rf CMakeOutput.log

run:
	./build/detect
