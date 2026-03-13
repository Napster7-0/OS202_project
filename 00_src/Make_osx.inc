CC = /opt/homebrew/bin/gcc-15
CXX = /opt/homebrew/bin/g++-15

SDL2_PREFIX = /opt/homebrew/opt/sdl2

LIB = -L$(SDL2_PREFIX)/lib -lSDL2 -lpthread -fopenmp