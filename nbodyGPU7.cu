// Same as GPU6 but but added EPSILON to take away check to see if you are not finding the force
// caused by yourself																																							
// nvcc nbodyGPU7.cu -o GPU7 -lglut -lm -lGLU -lGL
//To stop hit "control c" in the window you launched it from.
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 4096
#define BLOCK 256

#define XWindowSize 2500
#define YWindowSize 2500

#define DRAW 10
#define PRINT 100
#define DAMP 0.5

#define G 1.0
#define H 1.0
#define EPSILON 0.000001

#define DT 0.001
#define STOP_TIME 0.5

#define EYE 10.0
#define FAR 50.0

// Globals
float4 p[N];
float3 v[N], f[N];
float4 *p_GPU;
float3 *v_GPU, *f_GPU;
FILE *data_file, *data_file1, *data_file2;
dim3 block, grid;

void set_initail_conditions()
{
	int i,j,k,num,particles_per_side;
    float position_start, temp;
    float initail_seperation;
    
    if(N%BLOCK != 0)
	{
	    printf("\nError: Number of Particles is not a multiple of the block size \n\n");
	    exit(0);
    }

	temp = pow((float)N,1.0/3.0) + 0.99999;
	particles_per_side = temp;
	printf("\n cube root of N = %d \n", particles_per_side);
    position_start = -(particles_per_side -1.0)/2.0;
	initail_seperation = 2.0;
	
	for(i=0; i<N; i++)
	{
		p[i].w = 1.0;
	}
	
	num = 0;
	for(i=0; i<particles_per_side; i++)
	{
		for(j=0; j<particles_per_side; j++)
		{
			for(k=0; k<particles_per_side; k++)
			{
			    if(N <= num) break;
				p[num].x = position_start + i*initail_seperation;
				p[num].y = position_start + j*initail_seperation;
				p[num].z = position_start + k*initail_seperation;
				v[num].x = 0.0;
				v[num].y = 0.0;
				v[num].z = 0.0;
				num++;
			}
		}
	}
	
	block.x = BLOCK;
	block.y = 1;
	block.z = 1;
	
	grid.x = (N-1)/block.x + 1;
	grid.y = 1;
	grid.z = 1;
	
	cudaMalloc( (void**)&p_GPU, N *sizeof(float4) );
	cudaMalloc( (void**)&v_GPU, N *sizeof(float3) );
	cudaMalloc( (void**)&f_GPU, N *sizeof(float3) );
}

void draw_picture()
{
	int i;
	
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	glColor3d(1.0,1.0,0.5);
	for(i=0; i<N; i++)
	{
		glPushMatrix();
		glTranslatef(p[i].x, p[i].y, p[i].z);
		glutSolidSphere(0.1,20,20);
		glPopMatrix();
	}
	
	glutSwapBuffers();
}
                                 
__device__ float3 getBodyBodyForce(float4 p0, float4 p1)
{
    float3 f;
    float dx = p1.x - p0.x;
    float dy = p1.y - p0.y;
    float dz = p1.z - p0.z;
    float r2 = dx*dx + dy*dy + dz*dz + EPSILON;
    float r = sqrt(r2);
    
    float force  = (G*p0.w*p1.w)/(r2) - (H*p0.w*p1.w)/(r2*r2);
    
    f.x = force*dx/r;
    f.y = force*dy/r;
    f.z = force*dz/r;
    
    return(f);
}

__global__ void getForces(float4 *pos, float3 *vel, float3 * force)
{
    float3 force_mag, forceSum;
    float4 posMe;
    __shared__ float4 shPos[BLOCK];
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    
    	forceSum.x = 0.0;
		forceSum.y = 0.0;
		forceSum.z = 0.0;
		
		posMe.x = pos[id].x;
		posMe.y = pos[id].y;
		posMe.z = pos[id].z;
		posMe.w = pos[id].w;
	    
    	for(int j=0; j < gridDim.x; j++)
    	{
    			shPos[threadIdx.x] = pos[threadIdx.x + blockDim.x*j];
    			__syncthreads();
   
			#pragma unroll 32
        	for(int i=0; i < blockDim.x; i++)	
        	{
		    	force_mag = getBodyBodyForce(posMe, shPos[i]);
				forceSum.x += force_mag.x;
				forceSum.y += force_mag.y;
				forceSum.z += force_mag.z;
	   	 	}
		}
		force[id].x = forceSum.x;
		force[id].y = forceSum.y;
		force[id].z = forceSum.z;
}

__global__ void moveBodies(float4 *pos, float3 *vel, float3 * force)
{
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    if(id < N)
    {
	    vel[id].x += ((force[id].x-DAMP*vel[id].x)/pos[id].w)*DT;
	    vel[id].y += ((force[id].y-DAMP*vel[id].y)/pos[id].w)*DT;
	    vel[id].z += ((force[id].z-DAMP*vel[id].z)/pos[id].w)*DT;
	
	    pos[id].x += vel[id].x*DT;
	    pos[id].y += vel[id].y*DT;
	    pos[id].z += vel[id].z*DT;
    }
}

void n_body()
{
	float dt;
	int   tdraw = 0; 
	float time = 0.0;
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	dt = DT;
	
    cudaMemcpy( p_GPU, p, N *sizeof(float4), cudaMemcpyHostToDevice );
    cudaMemcpy( v_GPU, v, N *sizeof(float3), cudaMemcpyHostToDevice );
    
	while(time < STOP_TIME)
	{	
		getForces<<<grid, block>>>(p_GPU, v_GPU, f_GPU);
		moveBodies<<<grid, block>>>(p_GPU, v_GPU, f_GPU);

   
		if(tdraw == DRAW) 
		{
		    cudaMemcpy( p, p_GPU, N *sizeof(float4), cudaMemcpyDeviceToHost );
			draw_picture();
			tdraw = 0;
		}
		tdraw++;
	
		time += dt;
	}
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\n\nGPU time = %3.1f milliseconds\n", elapsedTime);
	cudaMemcpy( p, p_GPU, N *sizeof(float4), cudaMemcpyDeviceToHost );
}

void control()
{	
	set_initail_conditions();
	draw_picture();
    n_body();
    draw_picture();
	
	printf("\n DONE \n");
	while(1);
}

void Display(void)
{
	gluLookAt(EYE, EYE, EYE, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	control();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, FAR);

	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("2 Body 3D");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}






