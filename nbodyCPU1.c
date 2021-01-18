//General CPU code. Run on the upper trianglar part of the force matrix.
//Initail conditions are setup in a cube.																																												
// gcc nbodyCPU1.c -o CPU1 -lglut -lm -lGLU -lGL
//To stop hit "control c" in the window you launched it from.
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 100

#define XWindowSize 2500
#define YWindowSize 2500

#define DRAW 10
#define PRINT 100
#define DAMP .5

#define G 10.0
#define H 10.0

#define DT 0.0001

#define EYE 10.0
#define FAR 50.0

// Globals
float p[N][3], v[N][3], f[N][3], mass[N], STOP_TIME; 
FILE *data_file, *data_file1, *data_file2;

void set_initail_conditions()
{
	int i,j,k,num,particles_per_side;
    float position_start, temp;
    float initail_seperation, initial_cube_dimension;
	
	STOP_TIME = 100.0;

	temp = pow((float)N,1.0/3.0) + .99999;
	particles_per_side = temp;
	printf("\n N = %d cube root of N = %f particles per side = %d \n", N, (temp-0.9999), particles_per_side);
    position_start = -(particles_per_side -1.0)/2.0;
	initail_seperation = 2.0;
	
	for(i=0; i<N; i++)
	{
		mass[i] = 1.0;
	}
	
	num = 0;
	for(i=0; i<particles_per_side; i++)
	{
		for(j=0; j<particles_per_side; j++)
		{
			for(k=0; k<particles_per_side; k++)
			{
			    if(N <= num) break;
				p[num][0] = position_start + i*initail_seperation;
				p[num][1] = position_start + j*initail_seperation;
				p[num][2] = position_start + k*initail_seperation;
				v[num][0] = 0.0;
				v[num][1] = 0.0;
				v[num][2] = 0.0;
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
		glTranslatef(p[i][0], p[i][1], p[i][2]);
		glutSolidSphere(0.1,20,20);
		glPopMatrix();
	}
	
	glutSwapBuffers();
}

void n_body()
{
	float force_mag; 
	float dx,dy,dz,d, d2, dt;
	float dvx,dvy,dvz,close_seperate;
	int    tdraw = 0; 
	int    tprint = 0;
	float  time = 0.0;
	int i,j;
	
	dt = DT;

	while(time < STOP_TIME)
	{
		for(i=0; i<N; i++)
		{
			f[i][0] = 0.0;
			f[i][1] = 0.0;
			f[i][2] = 0.0;
		}
		
		for(i=0; i<N; i++)
		{
			for(j=i+1; j<N; j++)
			{
				dx = p[j][0]-p[i][0];
				dy = p[j][1]-p[i][1];
				dz = p[j][2]-p[i][2];
				d2 = dx*dx + dy*dy + dz*dz;
				d  = sqrt(d2);
				
				force_mag  = (G*mass[i]*mass[j])/(d2) - (H*mass[i]*mass[j])/(d2*d2);
				f[i][0] += force_mag*dx/d;
				f[j][0] -= force_mag*dx/d;
				f[i][1] += force_mag*dy/d;
				f[j][1] -= force_mag*dy/d;
				f[i][2] += force_mag*dz/d;
				f[j][2] -= force_mag*dz/d;
			}
		}

		for(i=0; i<N; i++)
		{
			if(time == 0.0)
			{
				v[i][0] += (f[i][0]/mass[i])*0.5*dt;
				v[i][1] += (f[i][1]/mass[i])*0.5*dt;
				v[i][2] += (f[i][2]/mass[i])*0.5*dt;
			}
			else
			{
				v[i][0] += ((f[i][0]-DAMP*v[i][0])/mass[i])*dt;
				v[i][1] += ((f[i][1]-DAMP*v[i][1])/mass[i])*dt;
				v[i][2] += ((f[i][2]-DAMP*v[i][2])/mass[i])*dt;
			}

			p[i][0] += v[i][0]*dt;
			p[i][1] += v[i][1]*dt;
			p[i][2] += v[i][2]*dt;
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
	int    tdraw = 0;
	float  time = 0.0;

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






