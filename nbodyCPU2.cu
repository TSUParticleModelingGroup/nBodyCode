//This is code write to run on the CPU but written just like you would write it for the GPU 
//and write to compiled with NVCC, It also uses the full force matrix.																																											
// nvcc nbodyCPU2.cu -o CPU2 -lglut -lm -lGLU -lGL
//To stop hit "control c" in the window you launched it from.
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 100

#define XWindowSize 2500
#define YWindowSize 2500

#define DRAW 10
#define PRINT 100
#define DAMP .5

#define G 1.0
#define H 1.0

#define DT 0.0001

#define EYE 10.0
#define FAR 50.0

// Globals
float4 p[N];
float3 v[N], f[N];
float STOP_TIME; 
FILE *data_file, *data_file1, *data_file2;

void set_initail_conditions()
{
	int i,j,k,num,particles_per_side;
    float position_start, temp;
    float initail_seperation;
	
	STOP_TIME = 100.0;
   
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
                                 
__host__ __device__ float3 getBodyBodyForce(float4 p0, float4 p1)
{
    float3 f;
    float dx = p1.x - p0.x;
    float dy = p1.y - p0.y;
    float dz = p1.z - p0.z;
    float r2 = dx*dx + dy*dy + dz*dz;
    float r = sqrt(r2);
    
    float force  = (G*p0.w*p1.w)/(r2) - (H*p0.w*p1.w)/(r2*r2);
    
    f.x = force*dx/r;
    f.y = force*dy/r;
    f.z = force*dz/r;
    
    return(f);
}

void n_body()
{
	float3 force_mag; 
	float dt;
	int   tdraw = 0; 
	int   tprint = 0;
	float time = 0.0;
	int i,j;
	
	dt = DT;

	while(time < STOP_TIME)
	{
		for(i=0; i<N; i++)
		{
			f[i].x = 0.0;
			f[i].y = 0.0;
			f[i].z = 0.0;
		}
		
		for(i=0; i<N; i++)
		{
			for(j=0; j<N; j++)
			{	
				if(i != j) 
				{
				    force_mag = getBodyBodyForce(p[i], p[j]);
				    f[i].x += force_mag.x;
				    f[i].y += force_mag.y;
				    f[i].z += force_mag.z;
				}
			}
		}

		for(i=0; i<N; i++)
		{
			if(time == 0.0)
			{
				v[i].x += (f[i].x/p[i].w)*0.5*dt;
				v[i].y += (f[i].y/p[i].w)*0.5*dt;
				v[i].z += (f[i].z/p[i].w)*0.5*dt;
			}
			else
			{
				v[i].x += ((f[i].x-DAMP*v[i].x)/p[i].w)*dt;
				v[i].y += ((f[i].y-DAMP*v[i].y)/p[i].w)*dt;
				v[i].z += ((f[i].z-DAMP*v[i].z)/p[i].w)*dt;
			}

			p[i].x += v[i].x*dt;
			p[i].y += v[i].y*dt;
			p[i].z += v[i].z*dt;
		}

		if(tdraw == DRAW) 
		{
			draw_picture();
			tdraw = 0;
		}
		
		time += dt;
		tdraw++;
		tprint++;
	}
}

void control()
{	
	set_initail_conditions();
	draw_picture();
    n_body();
	
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






