BUILD_DIR=.
SRC_DIR=src
CXX=g++
CXXFLAGS=-pipe -O2 -std=c++11
CXXSOFLAGS=-shared -fPIC -undefined dynamic_lookup
CXXWFLAGS=-Wall -Wextra -Werror
CXXDEBUGFLAGS=-g -std=c++11
INCLUDES=$(shell python -m pybind11 --includes)

py:
	g++ ${CXXFLAGS} ${CXXWFLAGS} ${CXXSOFLAGS} ${INCLUDES} \
		${SRC_DIR}/binding.cxx -o ${BUILD_DIR}/east.so

pydebug:
	g++ ${CXXDEBUGFLAGS} ${CXXWFLAGS} ${CXXSOFLAGS} ${INCLUDES} \
		${SRC_DIR}/binding.cxx -o ${BUILD_DIR}/east.so

bin:
	g++ ${CXXFLAGS} ${CXXWFLAGS} ${SRC_DIR}/main.cxx -o ${BUILD_DIR}/main

bindebug:
	g++ ${CXXDEBUGFLAGS} ${CXXWFLAGS} ${SRC_DIR}/main.cxx -o ${BUILD_DIR}/main

clean:
	rm -f main
	rm -f east.so
	rm -rf east.so.dSYM
	rm -rf main.dSYM

cleandirs:
	rm -f results/plots/*
	rm -f results/harvested/*
	rm -f results/pjk/*

cleandirspatch:
	rm -f results-patch/plots/*
	rm -f results-patch/harvested/*
	rm -f results-patch/pjk/*

tps:
	python analyze.py tps

tpspatch:
	python analyze.py patch-tps

histo:
	python analyze.py histo

histopatch:
	python analyze.py patch-histo
