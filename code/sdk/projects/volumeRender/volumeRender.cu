/*
* Copyright 1993-2008 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.  This source code is a "commercial item" as
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer software" and "commercial computer software
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*/

/*
    Volume rendering sample

    This sample loads a 3D volume from disk and displays it using
    ray marching and 3D textures.
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <GL/glew.h>

#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <cuda_gl_interop.h>
#include <cutil.h>

typedef unsigned int uint;
typedef unsigned char uchar;

#include <volumeRender_kernel.cu>

char *volumeFilename = "Bucky.raw";
cudaExtent volumeSize = make_cudaExtent(32, 32, 32);

uint width = 512, height = 512;
dim3 blockSize(16, 16);
dim3 gridSize(width / blockSize.x, height / blockSize.y);

float3 viewRotation;
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float invViewMatrix[12];

float density = 0.05f;
float brightness = 1.0f;
float transferOffset = 0.0f;
float transferScale = 1.0f;
bool linearFiltering = true;

cudaArray *d_volumeArray = 0;
cudaArray *d_transferFuncArray;
GLuint pbo = 0;     // OpenGL pixel buffer object

void initPixelBuffer();

// render image using CUDA
void render()
{
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeof(float4)*3) );

    // map PBO to get CUDA device pointer
    uint *d_output;
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&d_output, pbo));

    CUDA_SAFE_CALL(cudaMemset(d_output, 0, width*height*4));

    // call CUDA kernel, writing results to PBO
    d_render<<<gridSize, blockSize>>>(d_output, width, height, density, brightness, transferOffset, transferScale);
    CUT_CHECK_ERROR("kernel failed");

    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(pbo));
}

// display results using OpenGL (called by GLUT)
void display()
{
    // use OpenGL to build view matrix
    GLfloat modelView[16];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
        glLoadIdentity();
        glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
        glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
        glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glPopMatrix();

    invViewMatrix[0] = modelView[0]; invViewMatrix[1] = modelView[4]; invViewMatrix[2] = modelView[8]; invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1]; invViewMatrix[5] = modelView[5]; invViewMatrix[6] = modelView[9]; invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2]; invViewMatrix[9] = modelView[6]; invViewMatrix[10] = modelView[10]; invViewMatrix[11] = modelView[14];

    render();

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glutSwapBuffers();
    glutReportErrors();
}

void idle()
{
}

void keyboard(unsigned char key, int x, int y)
{
    switch(key) {
        case 27:
            exit(0);
            break;
        case 'f':
            linearFiltering = !linearFiltering;
            tex.filterMode = linearFiltering ? cudaFilterModeLinear : cudaFilterModePoint;
            break;
        case '=':
            density += 0.01;
            break;
        case '-':
            density -= 0.01;
            break;
        case '+':
            density += 0.1;
            break;
        case '_':
            density -= 0.1;
            break;

        case ']':
            brightness += 0.1;
            break;
        case '[':
            brightness -= 0.1;
            break;

        case ';':
            transferOffset += 0.01;
            break;
        case '\'':
            transferOffset -= 0.01;
            break;

        case '.':
            transferScale += 0.01;
            break;
        case ',':
            transferScale -= 0.01;
            break;

        default:
            break;
    }
    printf("density = %.2f, brightness = %.2f, transferOffset = %.2f, transferScale = %.2f\n", density, brightness, transferOffset, transferScale);
    glutPostRedisplay();
}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
        buttonState |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    ox = x; oy = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - ox;
    dy = y - oy;

    if (buttonState == 3) {
        // left+middle = zoom
        viewTranslation.z += dy / 100.0;
    } 
    else if (buttonState & 2) {
        // middle = translate
        viewTranslation.x += dx / 100.0;
        viewTranslation.y -= dy / 100.0;
    }
    else if (buttonState & 1) {
        // left = rotate
        viewRotation.x += dy / 5.0;
        viewRotation.y += dx / 5.0;
    }

    ox = x; oy = y;
    glutPostRedisplay();
}

void reshape(int x, int y)
{
    width = x; height = y;
    initPixelBuffer();

    glViewport(0, 0, x, y);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 
}

void initCuda(uchar *h_volume, cudaExtent volumeSize)
{
    // create 3D array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar>();
    CUDA_SAFE_CALL( cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize) );

    // The sample does not work with pageable memory
    // This is a known issue for beta that will be fixed for the public release
#define USE_PAGE_LOCKED_MEMORY 1
#if USE_PAGE_LOCKED_MEMORY
    // copy to page-locked mem
    cudaPitchedPtr pagelockedPtr;
    pagelockedPtr.pitch = volumeSize.width*sizeof(uchar);
    pagelockedPtr.xsize = volumeSize.width;
    pagelockedPtr.ysize = volumeSize.height;
    size_t size = volumeSize.width*volumeSize.height*volumeSize.depth*sizeof(uchar);
    CUDA_SAFE_CALL( cudaMallocHost(&(pagelockedPtr.ptr), size) );
    memcpy(pagelockedPtr.ptr, h_volume, size);
#endif

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
#if USE_PAGE_LOCKED_MEMORY
    copyParams.srcPtr   = pagelockedPtr;
#else
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)h_volume, volumeSize.width*sizeof(uchar), volumeSize.width, volumeSize.height);
#endif
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    CUDA_SAFE_CALL( cudaMemcpy3D(&copyParams) );  

    // set texture parameters
    tex.normalized = true;                      // access with normalized texture coordinates
    tex.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex.addressMode[0] = cudaAddressModeClamp;  // wrap texture coordinates
    tex.addressMode[1] = cudaAddressModeClamp;

    // bind array to 3D texture
    CUDA_SAFE_CALL(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));

    // create transfer function texture
    float4 transferFunc[] = {
        {  0.0, 0.0, 0.0, 0.0, },
        {  1.0, 0.0, 0.0, 1.0, },
        {  1.0, 0.5, 0.0, 1.0, },
        {  1.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 1.0, 1.0, },
        {  0.0, 0.0, 1.0, 1.0, },
        {  1.0, 0.0, 1.0, 1.0, },
        {  0.0, 0.0, 0.0, 0.0, },
    };

    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
    cudaArray* d_transferFuncArray;
    CUDA_SAFE_CALL(cudaMallocArray( &d_transferFuncArray, &channelDesc2, sizeof(transferFunc)/sizeof(float4), 1)); 
    CUDA_SAFE_CALL(cudaMemcpyToArray( d_transferFuncArray, 0, 0, transferFunc, sizeof(transferFunc), cudaMemcpyHostToDevice));

    transferTex.filterMode = cudaFilterModeLinear;
    transferTex.normalized = true;    // access with normalized texture coordinates
    transferTex.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates

    // Bind the array to the texture
    CUDA_SAFE_CALL( cudaBindTextureToArray( transferTex, d_transferFuncArray, channelDesc2));
}

void cleanup()
{
    CUDA_SAFE_CALL(cudaFreeArray(d_volumeArray));
    CUDA_SAFE_CALL(cudaFreeArray(d_transferFuncArray));
	CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(pbo));    
	glDeleteBuffersARB(1, &pbo);
}

int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void initPixelBuffer()
{
    if (pbo) {
        // delete old buffer
        CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(pbo));
        glDeleteBuffersARB(1, &pbo);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	CUDA_SAFE_CALL(cudaGLRegisterBufferObject(pbo));

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));
}

// Load raw data from disk
uchar *loadRawFile(char *filename, size_t size)
{
	FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

	uchar *data = (uchar *) malloc(size);
	size_t read = fread(data, 1, size, fp);
	fclose(fp);

    printf("Read '%s', %d bytes\n", filename, read);

    return data;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    CUT_DEVICE_INIT(argc, argv);

    // parse arguments
    char *filename;
    if (cutGetCmdLineArgumentstr( argc, (const char**) argv, "file", &filename)) {
        volumeFilename = filename;
    }
    int n;
    if (cutGetCmdLineArgumenti( argc, (const char**) argv, "size", &n)) {
        volumeSize.width = volumeSize.height = volumeSize.depth = n;
    }
    if (cutGetCmdLineArgumenti( argc, (const char**) argv, "xsize", &n)) {
        volumeSize.width = n;
    }
    if (cutGetCmdLineArgumenti( argc, (const char**) argv, "ysize", &n)) {
        volumeSize.height = n;
    }
    if (cutGetCmdLineArgumenti( argc, (const char**) argv, "zsize", &n)) {
         volumeSize.depth = n;
    }

    // load volume data
    char* path = cutFindFilePath(volumeFilename, argv[0]);
    if (path == 0) {
        fprintf(stderr, "Error finding file '%s'\n", volumeFilename);
        exit(EXIT_FAILURE);
    }

    size_t size = volumeSize.width*volumeSize.height*volumeSize.depth;
    uchar *h_volume = loadRawFile(path, size);
    initCuda(h_volume, volumeSize);
    free(h_volume);

    printf("Press '=' and '-' to change density\n"
           "      ']' and '[' to change brightness\n"
           "      ';' and ''' to modify transfer function offset\n"
           "      '.' and ',' to modify transfer function scale\n");

    // initialize GLUT callback functions
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA volume rendering");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(-1);
    }
    initPixelBuffer();

    atexit(cleanup);

    glutMainLoop();
    return 0;
}
