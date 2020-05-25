#include <iostream>
#include <string>
#include <math.h>
#include <easyppm.h>
using namespace std;

class Tuple {
    public:
        float x;
        float y;
        float z;
        float w;

        __host__ __device__
        Tuple() {}

        __host__ __device__
        Tuple(float xpos, float ypos, float zpos, float weight) {
            x = xpos;
            y = ypos;
            z = zpos;
            w = weight;
        }

        __host__ __device__
        Tuple operator + (Tuple const &t1) {
            Tuple res;
            res.x = t1.x + x;
            res.y = t1.y + y;
            res.z = t1.z + z;
            res.w = t1.w + w;
            return res;
        }

        __host__ __device__
        Tuple operator - (Tuple const &t1) {
            Tuple res;
            res.x = x - t1.x;
            res.y = y - t1.y;
            res.z = z - t1.z;
            res.w = w - t1.w;
            return res;
        }

        __host__ __device__
        Tuple operator * (float const &mult) {
            Tuple res;
            res.x = mult * x;
            res.y = mult * y;
            res.z = mult * z;
            res.w = mult * w;
            return res;
        }

        __host__ __device__
        Tuple operator / (float const &div) {
            Tuple res;
            res.x = x/div;
            res.y = y/div;
            res.z = z/div;
            res.w = w/div;
            return res;
        }

        __host__ __device__
        Tuple neg() {
            Tuple res;
            res.x = -x;
            res.y = -y;
            res.z = -z;
            res.w = -w;
            return res;
        }

        __host__ __device__
        float mag() {
            float res = sqrt( x*x  + y*y  + z*z  + w*w );
            return res;
        }

        __host__ __device__
        Tuple norm() {
            Tuple res;
            float magnitude = mag();
            res.x = x/magnitude;
            res.y = y/magnitude;
            res.z = z/magnitude;
            res.w = w/magnitude;
            return res;
        }

        __host__ __device__
        static float dot(Tuple t1, Tuple t2) {
            float res = t1.x*t2.x + t1.y*t2.y + t1.z*t2.z + t1.w*t2.w;
            return res;
        }

        __host__ __device__
        static Tuple cross(Tuple t1, Tuple t2) {
            Tuple res;
            res.x = t1.y*t2.z + t1.z*t2.y;
            res.y = t1.z*t2.x + t1.x*t2.z;
            res.z = t1.x*t2.y + t1.y*t2.x;
            res.w = 0.0;
            return res;
        }

        __host__ __device__
        static bool equalsFloat(float f1, float f2) {
            float EPSILON = 0.00001;
            if(abs(f1-f2) < EPSILON) {
                return true;
            } else {
                return false;
            }
        }

        __host__ __device__
        static bool equals(Tuple t1, Tuple t2) {
            if( equalsFloat(t1.x,t2.x) && equalsFloat(t1.y,t2.y) && equalsFloat(t1.z,t2.z) && equalsFloat(t1.w,t2.w) ) { 
                return true;
            } else {
                return false;
            }
        }

};

class Matrix {
    public:
        __host__ __device__
        static bool equals(float *M1, float *M2, int size) {
            for(int i = 0; i < size; i++) {
                for(int j = 0; j < size; j++) {
                    if(!Tuple::equalsFloat(M1[size*i+j], M2[size*i+j])) {
                        return false;
                    }
                }
            }
            return true;
        }

        __host__ __device__
        static void multiply(float *M1, float *M2, float *M3) {
            for(int i = 0; i < 4; i++) {
                for(int j = 0; j < 4; j++) {
                    M3[4*i+j] = M1[4*i+0] * M2[4*0+j] + M1[4*i+1] * M2[4*1+j] + M1[4*i+2] * M2[4*2+j] + M1[4*i+3] * M2[4*3+j]; 
                }
            }
        }

        __host__ __device__
        static Tuple tupleMultiply(float *M, Tuple t) {
            Tuple res;
            res.x = M[0] * t.x + M[1] * t.y + M[2] * t.z + M[3] * t.w;
            res.y = M[4] * t.x + M[5] * t.y + M[6] * t.z + M[7] * t.w;
            res.z = M[8] * t.x + M[9] * t.y + M[10] * t.z + M[11] * t.w;
            res.w = M[12] * t.x + M[13] * t.y + M[14] * t.z + M[15] * t.w;
            return res;
        }

        __host__ __device__
        static void transpose(float *M1, float *M2) {
            for(int i = 0; i < 4; i++) {
                for(int j = 0; j < 4; j++) {
                    M2[4*j+i] = M1[4*i+j]; 
                }
            }
        }

        __host__ __device__
        static float determinant2x2(float *M) {
            return M[0]*M[3] - M[1]*M[2];
        }

        __host__ __device__
        static void submatrix4x4(float *M, int row, int col, float* out) {
            int countRow = 0;
            int countCol = 0;
            for(int i = 0; i < 3; i++) {
                for(int j = 0; j < 3; j++) { 
                    if(i>=row) { countRow = 1; }
                    if(j>=col) { countCol = 1; }
                    out[3*i+j] = M[4*i+4*countRow+j+countCol];
                    countRow = 0;
                    countCol = 0;
                }
            }
        }

        __host__ __device__
        static void submatrix3x3(float *M, int row, int col, float* out) {
            int countRow = 0;
            int countCol = 0;
            for(int i = 0; i < 2; i++) {
                for(int j = 0; j < 2; j++) { 
                    if(i>=row) { countRow = 1; }
                    if(j>=col) { countCol = 1; }
                    out[2*i+j] = M[3*i+3*countRow+j+countCol];
                    countRow = 0;
                    countCol = 0;
                }
            }
        }

        __host__ __device__
        static float minor3x3(float *M, int row, int col) {
            float sub[4];
            submatrix3x3(M, row, col, sub);
            float det = determinant2x2(sub);
            return det;
        }

        __host__ __device__
        static float cofactor3x3(float *M, int row, int col) {
            float minor = minor3x3(M, row, col);
            if((row+col)%2==0) {
                return minor;
            } else {
                return -1*minor;
            }
        }

        __host__ __device__
        static float determinant3x3(float *M) {
            float det;
            for(int i = 0; i < 3; i++) {
                det = det + M[i] * cofactor3x3(M,0,i);
            }
            return det;
        }

        __host__ __device__
        static float minor4x4(float *M, int row, int col) {
            float sub[9];
            submatrix4x4(M, row, col, sub);
            float det = determinant3x3(sub);
            return det;
        }

        __host__ __device__
        static float cofactor4x4(float *M, int row, int col) {
            float minor = minor4x4(M, row, col);
            if((row+col)%2==0) {
                return minor;
            } else {
                return -1*minor;
            }
        }

        __host__ __device__
        static float determinant(float *M) {
            float det;
            for(int i = 0; i < 4; i++) {
                det = det + M[i] * cofactor4x4(M,0,i);
            }
            return det;
        }

        __host__ __device__
        static void inverse(float *M, float *out) {
            float det = determinant(M);
            if(det == 0) { return; }

            for(int i = 0; i<4; i++) {
                for(int j = 0; j<4; j++) {
                    float c = cofactor4x4(M,i,j);
                    out[4*j+i] = c/det;
                }
            }
        }

};

class Ray {
    public:
        Tuple origin;
        Tuple direction;

        __host__ __device__
        Ray(Tuple o, Tuple d) {
            origin = o;
            direction = d;
        }
        
        __host__ __device__
        static Tuple position(Ray r, float t) {
            return r.origin + r.direction * t;
        }
};

class Sphere {
    public:
        Tuple origin;
        float radius;

        __host__ __device__
        Sphere(Tuple o, float r) {
            origin = o;
            radius = r;
        }
};

__host__ __device__
void intersect(Ray r, Sphere s, float *out) {
    Tuple sphere_to_ray = r.origin - s.origin;

    float a = Tuple::dot(r.direction, r.direction);
    float b = 2 * Tuple::dot(r.direction, sphere_to_ray);
    float c = Tuple::dot(sphere_to_ray, sphere_to_ray) - 1;

    float discriminant = b*b - 4 * a * c;

    if(discriminant < 0) {
        out[0] = 0;
        return;
    }
    out[0] = 2;
    out[1] = (-b - sqrt(discriminant)) / (2 * a);
    out[2] = (-b + sqrt(discriminant)) / (2 * a);
}

__host__ __device__
void hit(float *i, int *color) {
    if(i[0] ==0) {
        return;
    }
    float fraction = abs(i[1]-i[2])/2;
    float scale = fraction*0.75 + 0.25;
    float c = 255 * scale;
    color[0] = (int) c;

}

void softwareTracer(int width, Tuple rOrigin, Sphere s, float wallZ, float *out) {
    float wall_size = 10;
    float pixel_size = wall_size/width;
    float half = wall_size/2;
    for(int i = 0; i < width; i++) {
        float y = half - pixel_size * i;
        for(int j = 0; j < width; j++) {
            float x = -1*half + pixel_size * j;

            Tuple pixel(x,y,wallZ,1.0);
            Tuple dist = pixel - rOrigin;
            Tuple rDirection = dist.norm();
            Ray r(rOrigin, rDirection);
            float o[3];
            intersect(r,s,o);
            int color[1];
            hit(o, color);
            out[width*i+j] = color[0];
        }
    }
}

__global__ void hardwareTracer(int width, Tuple rOrigin, Sphere s, float wallZ, float *out) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    if(i<width && j<<width) {
        float wall_size = 10;
        float pixel_size = wall_size/width;
        float half = wall_size/2;
        
        float y = half - pixel_size * i;

        float x = -1*half + pixel_size * j;

        Tuple pixel(x,y,wallZ,1.0);
        Tuple dist = pixel - rOrigin;
        Tuple rDirection = dist.norm();
        Ray r(rOrigin, rDirection);
        float o[3];
        intersect(r,s,o);
        int color[1];
        hit(o, color);
        out[width*i+j] = color[0];
    }
}

int main(int argc, char **argv) {
    bool write_image = false;
    int width = 9000;
    float *hostOutput;
    float *deviceOutput;

    hostOutput = (float *) malloc(width*width*sizeof(float));
    Tuple rayOrigin(0.0, 0.0, -5.0, 1.0);
    Tuple sphereOrigin(0.0, 0.0, 0.0, 1.0);
    Sphere sphere(sphereOrigin, 1.0);
    float wallZ = 15.0;

    // Software Ray Tracer
    const clock_t begin_time = clock();
    softwareTracer(width, rayOrigin, sphere, wallZ, hostOutput);
    cout << float(clock() - begin_time) / CLOCKS_PER_SEC << "\n";

    // Hardware Ray Tracer
    const clock_t begin_time1 = clock();
    cudaMalloc((void **)&deviceOutput, width*width*sizeof(float));

    dim3 dimGrid(ceil((width)/16.0), ceil((width)/16.0), 1);
    dim3 dimBlock(16, 16, 1);

    hardwareTracer<<<dimGrid, dimBlock>>>(width, rayOrigin, sphere, wallZ, deviceOutput);
    
    cudaMemcpy(hostOutput, deviceOutput, width*width*sizeof(float), cudaMemcpyDeviceToHost);
    cout << float(clock() - begin_time1) / CLOCKS_PER_SEC << "\n";

    //write image to file
    if(write_image) {
        PPM ppm = easyppm_create(width,width,IMAGETYPE_PPM);
        easyppm_clear(&ppm, easyppm_rgb(0, 0, 0));
        for(int i = 0; i<width; i++) {
            for(int j = 0; j<width; j++) {
                easyppm_set(&ppm, i, j, easyppm_rgb(hostOutput[width*i+j], 0, 0));
            }
        }
        easyppm_write(&ppm, "image.ppm");
    }

    free(hostOutput);
    cudaFree(deviceOutput);

    cout << endl;
    
    return 0;
}


