CC=g++
DEBUG=-O3 
CXX_FLAGS=-std=c++11 -DCPU_ONLY 
LD_FLAGS=-I /home/atsg/PycharmProjects/gvh205/caffe_1.0/include/ -I /home/atsg/PycharmProjects/gvh205/caffe_1.0/build/src/ -L /home/atsg/PycharmProjects/gvh205/caffe_1.0/build/lib/ -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_video -lopencv_videoio -lcaffe -lglog -lboost_system -lprotobuf 
TARGETS=video picture fddb pcn_api_test
LIB=libPCN.so
LIB_DIR=/usr/local/lib
SHARE_DIR=/usr/local/share/pcn

all: $(TARGETS)

$(LIB): PCN.cpp
	$(CC) -shared -fPIC -o $@ $^ ${CXX_FLAGS} ${LD_FLAGS}  ${DEBUG}

$(TARGETS): $(LIB)
	$(CC) -o $@ $@.cpp $(LIB) -std=c++11 -D CPU_ONLY ${LD_FLAGS} ${DEBUG} 

install:
	mkdir -p $(SHARE_DIR)
	cp model/* $(SHARE_DIR)
	cp $(LIB) $(LIB_DIR)

clean:
	rm -rf $(TARGETS) $(LIB)

