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
    Bicubic texture filtering sample
    sgreen 6/2008

    This sample demonstrates how to efficiently implement bicubic texture
    filtering in CUDA.

    Bicubic filtering is a higher order interpolation method that produces
    smoother results than bilinear interpolation:
    http://en.wikipedia.org/wiki/Bicubic

    It requires reading a 4 x 4 pixel neighbourhood rather than the
    2 x 2 area required by bilinear filtering.

    Current graphics hardware doesn't support bicubic filtering natively,
    but it is possible to compose a bicubic filter using just 4 bilinear
    lookups by offsetting the sample position within each texel and weighting
    the samples correctly. The only disadvantage to this method is that the
    hardware only maintains 9-bits of filtering precision within each texel.
    
    See "Fast Third-Order Texture Filtering", Sigg & Hadwiger, GPU Gems 2:
    http://developer.nvidia.com/object/gpu_gems_2_home.html
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

#include <bicubicTexture_kernel.cu>

char *imageFilename = "lena_bw.pgm";

uint width = 512, height = 512;
uint imageWidth, imageHeight;
dim3 blockSize(16, 16);
dim3 gridSize(width / blockSize.x, height / blockSize.y);

enum Mode { MODE_NEAREST, MODE_BILINEAR, MODE_BICUBIC };
Mode mode = MODE_BICUBIC;

cudaArray *d_imageArray = 0;
GLuint pbo = 0;                 // OpenGL pixel buffer object
GLuint displayTex = 0;

float tx = 9.0f, ty = 10.0f;    // image translation
float scale = 1.0f / 16.0f;     // image scale
float cx, cy;                   // image centre

void initPixelBuffer();
void runBenchmark(int iterations);

// render image using CUDA
void render(uchar *output)
{
    // call CUDA kernel, writing results to PBO memory
    switch(mode) {
    case MODE_NEAREST:
        tex.filterMode = cudaFilterModePoint;
        d_render<<<gridSize, blockSize>>>(output, width, height, tx, ty, scale, cx, cy);
        break;
    case MODE_BILINEAR:
        tex.filterMode = cudaFilterModeLinear;
        d_render<<<gridSize, blockSize>>>(output, width, height, tx, ty, scale, cx, cy);
        break;
    case MODE_BICUBIC:
        tex.filterMode = cudaFilterModeLinear;
        d_renderBicubic<<<gridSize, blockSize>>>(output, width, height, tx, ty, scale, cx, cy);
        break;
    }
    CUT_CHECK_ERROR("kernel failed");
}

// display results using OpenGL (called by GLUT)
void display()
{
    // map PBO to get CUDA device pointer
    uchar *d_output;
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&d_output, pbo));
    render(d_output);
    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(pbo));

    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // download image from PBO to OpenGL texture
    glDisable(GL_DEPTH_TEST);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, displayTex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_LUMINANCE, GL_UNSIGNED_BYTE, 0);

    // draw textured quad
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glVertex2f(0, 0); glTexCoord2f(0, 0);
    glVertex2f(0, 1); glTexCoord2f(1, 0);
    glVertex2f(1, 1); glTexCoord2f(1, 1);
    glVertex2f(1, 0); glTexCoord2f(0, 1);
    glEnd();
    glDisable(GL_TEXTURE_2D);

    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glutSwapBuffers();
    glutReportErrors();
}

// GLUT callback functions
void idle()
{
    glutPostRedisplay();
}

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch(key) {
        case 27:
            exit(0);
            break;
        case '1':
            mode = MODE_NEAREST;
            break;
        case '2':
            mode = MODE_BILINEAR;
            break;
        case '3':
            mode = MODE_BICUBIC;
            break;

        case '=':
        case '+':
            scale *= 0.5f;
            break;
        case '-':
            scale *= 2.0f;
            break;
        case 'r':
            scale = 1.0f;
            tx = ty = 0.0f;
            break;
        case 'd':
            printf("%f, %f, %f\n", tx, ty, scale);
        case 'b':
            runBenchmark(500);
            break;
        default:
            break;
    }

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

    if (buttonState & 1) {
        // left = translate
        tx -= dx*scale;
        ty -= dy*scale;
    }
    else if (buttonState & 2) {
        // middle = zoom
        scale -= dy / 1000.0;
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

void cleanup()
{
    CUDA_SAFE_CALL(cudaFreeArray(d_imageArray));
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
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte), 0, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	CUDA_SAFE_CALL(cudaGLRegisterBufferObject(pbo));

    // create texture for display
    if (displayTex) {
        glDeleteTextures(1, &displayTex);
    }
    glGenTextures(1, &displayTex);
    glBindTexture(GL_TEXTURE_2D, displayTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE8, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));
}

void mainMenu(int i)
{
    keyboard(i, 0, 0);
}

void initMenus()
{
    glutCreateMenu(mainMenu);
    glutAddMenuEntry("Nearest filtering [1]", '1');
    glutAddMenuEntry("Bilinear filtering [2]", '2');
    glutAddMenuEntry("Bicubic filtering [3]", '3');
    glutAddMenuEntry("Zoom in [=]", '=');
    glutAddMenuEntry("Zoom out [-]", '-');
    glutAddMenuEntry("Benchmark [b]", 'b');
    glutAddMenuEntry("Quit [esc]", 27);
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void runBenchmark(int iterations)
{
    unsigned int timer;
    CUT_SAFE_CALL(cutCreateTimer(&timer));

    uchar *d_output;
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&d_output, pbo));

    CUT_SAFE_CALL(cutStartTimer(timer));  
    for (int i = 0; i < iterations; ++i)
    {
        render(d_output);
    }

    cudaThreadSynchronize();
    CUT_SAFE_CALL(cutStopTimer(timer));  
    float time = cutGetTimerValue(timer) / (float) iterations;

    CUDA_SAFE_CALL(cudaGLUnmapBufferObject(pbo));

    printf("time: %0.3f ms, %f Mpixels/sec\n", time, (width*height / (time * 0.001f)) / 1e6);    
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
        imageFilename = filename;
    }

    bool benchmark = cutCheckCmdLineFlag(argc, (const char**) argv, "benchmark") != 0;

    // load image from disk
    uchar* h_data = NULL;
    char* imagePath = cutFindFilePath(imageFilename, argv[0]);
    if (imagePath == 0)
        exit(EXIT_FAILURE);
    CUT_SAFE_CALL(cutLoadPGMub(imagePath, &h_data, &imageWidth, &imageHeight));

    printf("Loaded '%s', %d x %d pixels\n", imageFilename, imageWidth, imageHeight);

    cx = imageWidth * 0.5f;
    cy = imageHeight * 0.5f;

    // allocate array and copy image data
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    CUDA_SAFE_CALL( cudaMallocArray(&d_imageArray, &channelDesc, imageWidth, imageHeight) ); 
    uint size = imageWidth * imageHeight * sizeof(uchar);
    CUDA_SAFE_CALL( cudaMemcpyToArray(d_imageArray, 0, 0, h_data, size, cudaMemcpyHostToDevice) );
    cutFree(h_data);

    // set texture parameters
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = false;    // access with integer texture coordinates

    // Bind the array to the texture
    CUDA_SAFE_CALL( cudaBindTextureToArray(tex, d_imageArray, channelDesc) );

    printf(
        "Press '=' and '-' to zoom\n"
        "Press number keys to change filtering mode:\n"
        "1 - nearest filtering\n"
        "2 - bilinear filtering\n"
        "3 - bicubic filtering\n"
        );


    // initialize GLUT callback functions
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA bicubic texture filtering");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    initMenus();

    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "Required OpenGL extensions missing.");
        exit(-1);
    }
    initPixelBuffer();

    atexit(cleanup);

    if (benchmark) {
        runBenchmark(500);
        exit(0);
    }

    glutMainLoop();
    return 0;
}
