all:
	nvcc raytracer.cu easyppm.c -I./ -arch=sm_35
