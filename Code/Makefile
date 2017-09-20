INCLUDE_DIRS = 
LIB_DIRS = 
CC=g++

CDEFS=
CFLAGS= -O0 -pg -g --std=c++11 $(INCLUDE_DIRS) $(CDEFS)
LIBS= -lrt -lpthreads
CPPLIBS= -L/usr/lib -lopencv_core -lopencv_flann -lopencv_video -lrt -lpthread

HFILES= /home/pi/test_Proj/include
CFILES= 
CPPFILES= capture.cpp test_videoCapture.cpp

SRCS= ${HFILES} ${CFILES}
CPPOBJS= ${CPPFILES:.cpp=.o}

all:	capture test_videoCapture

clean:
	-rm -f *.o *.d
	-rm -f capture test_videoCapture

distclean:
	-rm -f *.o *.d

capture: capture.o
	$(CC) $(LDFLAGS) $(CFLAGS) -I$(HFILES) -o $@ $@.o `pkg-config --libs opencv` $(CPPLIBS)  

test_videoCapture: test_videoCapture.o
	$(CC) $(LDFLAGS) $(CFLAGS) -I$(HFILES) -o $@ $@.o `pkg-config --libs opencv` $(CPPLIBS)
depend:

.c.o:
	$(CC) $(CFLAGS) -c $<

.cpp.o:
	$(CC) $(CFLAGS) -c $<
