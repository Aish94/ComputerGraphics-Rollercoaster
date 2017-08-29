/*
  CSCI 420 Computer Graphics, USC
  Assignment 2: Roller Coaster
  C++ starter code

  Student username: averghes
*/
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <cstring>
#include <sstream>
#include <iomanip>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "glm/ext.hpp"

#include "openGLHeader.h"
#include "glutHeader.h"
#include "imageIO.h"
#include "openGLMatrix.h"
#include "basicPipelineProgram.h"

using namespace std;

vector<float> rail;
vector<float> plank;

int numOfVertices;
float s = 0.5;
int frame;
int iteration = 0;
int screenshot_iteration = 0;
int prev_time = 0;
int time_now;

OpenGLMatrix* spline_matrix, *skybox_matrix;
BasicPipelineProgram * pipelineProgram, * skybox_pipelineProgram;
GLuint program, skybox_program;

GLuint rail_vao,rail_vbo,plank_vao,plank_vbo,skybox_vao,skybox_vbo,texture;

glm::fvec3* P;
glm::fvec3* T;
glm::fvec3* N;
glm::fvec3* B;

glm::vec3 cameraPos, targetPos,upVec;
int windowWidth = 1280;
int windowHeight = 720;
char windowTitle[512] = "CSCI 420 homework II";

// state of the world
float max_coord = 0.0f;
GLfloat skybox_vertices[108];

/*RLTDBF*/
const GLchar* texture_faces[] = {
  "ame_siege/siege_bk.tga", //Back face
  "ame_siege/siege_ft.tga", //Front Face
  "ame_siege/siege_up.tga", //Up Face
  "ame_siege/siege_dn.tga", //Down face
  "ame_siege/siege_lf.tga", //Left Face
  "ame_siege/siege_rt.tga", //Right Face
    };

// represents one control point along the spline 
struct Point 
{
  float x;
  float y;
  float z;
};
vector<Point> camera_points;
// spline struct 
// contains how many control points the spline has, and an array of control points 
struct Spline 
{
  int numControlPoints;
  Point * points;
};

struct cameraParams
{
  Point pos;
  Point focus;
  Point up;
};
vector<cameraParams> camera;

// the spline array 
Spline * splines;
// total number of splines 
int numSplines;

// write a screenshot to the specified filename
void saveScreenshot(string filename)
{
  unsigned char * screenshotData = new unsigned char[windowWidth * windowHeight * 3];
  glReadPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_UNSIGNED_BYTE, screenshotData);

  ImageIO screenshotImg(windowWidth, windowHeight, 3, screenshotData);

  string path = "screenshots/"+filename;  //save in screenshot folder
  if (screenshotImg.save(path.c_str(), ImageIO::FORMAT_JPEG) == ImageIO::OK)
    cout << "File " << filename << " saved successfully." << endl;
  else cout << "Failed to save file " << filename << '.' << endl;

  delete [] screenshotData;
}

//Add rail vertices to vector
void addRail(Point A, Point B, Point C)
{
 rail.push_back(A.x); 
 rail.push_back(A.y); 
 rail.push_back(A.z);

 rail.push_back(B.x); 
 rail.push_back(B.y); 
 rail.push_back(B.z);

 rail.push_back(C.x); 
 rail.push_back(C.y); 
 rail.push_back(C.z);
} 

//Add Plank vertices to vector
void addPlank(Point A, Point B, Point C)
{
   plank.push_back(A.x); 
   plank.push_back(A.y); 
   plank.push_back(A.z);

   plank.push_back(B.x); 
   plank.push_back(B.y); 
   plank.push_back(B.z);

   plank.push_back(C.x); 
   plank.push_back(C.y); 
   plank.push_back(C.z);
}

float alpha = 0.1;  //Thickness of rail
void genRails(int i)
{
  
  Point V0,V1,V2,V3,V4,V5,V6,V7;
  Point camera_point;

  //Back face
  V0.x = P[i].x + alpha*(-N[i].x + B[i].x);
  V0.y = P[i].y + alpha*(-N[i].y + B[i].y);
  V0.z = P[i].z + alpha*(-N[i].z + B[i].z);

  V1.x = P[i].x + alpha*(N[i].x + B[i].x);
  V1.y = P[i].y + alpha*(N[i].y + B[i].y);
  V1.z = P[i].z + alpha*(N[i].z + B[i].z);

  V2.x = P[i].x + alpha*(N[i].x - B[i].x);
  V2.y = P[i].y + alpha*(N[i].y - B[i].y);
  V2.z = P[i].z + alpha*(N[i].z - B[i].z);

  V3.x = P[i].x + alpha*(-N[i].x - B[i].x);
  V3.y = P[i].y + alpha*(-N[i].y - B[i].y);
  V3.z = P[i].z + alpha*(-N[i].z - B[i].z);

  //Front face
  V4.x = P[i+1].x + alpha*(-N[i+1].x + B[i+1].x);
  V4.y = P[i+1].y + alpha*(-N[i+1].y + B[i+1].y);
  V4.z = P[i+1].z + alpha*(-N[i+1].z + B[i+1].z);

  V5.x = P[i+1].x + alpha*(N[i+1].x + B[i+1].x);
  V5.y = P[i+1].y + alpha*(N[i+1].y + B[i+1].y);
  V5.z = P[i+1].z + alpha*(N[i+1].z + B[i+1].z);

  V6.x = P[i+1].x + alpha*(N[i+1].x - B[i+1].x);
  V6.y = P[i+1].y + alpha*(N[i+1].y - B[i+1].y);
  V6.z = P[i+1].z + alpha*(N[i+1].z - B[i+1].z);

  V7.x = P[i+1].x + alpha*(-N[i+1].x - B[i+1].x);
  V7.y = P[i+1].y + alpha*(-N[i+1].y - B[i+1].y);
  V7.z = P[i+1].z + alpha*(-N[i+1].z - B[i+1].z);

  //Every 20 steps add the plank
  if(i%20 == 0)
  {
    //Front face
    addPlank(V4,V5,V6);
    addPlank(V4,V7,V6);

    //Top face
    addPlank(V1,V2,V6);
    addPlank(V1,V5,V6);

    //Bottom face
    addPlank(V0,V3,V4);
    addPlank(V4,V7,V3);
  }

  //Rails at every point
  //Left face
  addRail(V4,V1,V0);
  addRail(V4,V1,V5);

  //Right face
  addRail(V2,V3,V7);
  addRail(V2,V7,V6);
}

void keyboardFunc(unsigned char key, int x, int y)
{
  switch (key)
  {
    case 27: // ESC key
      // Properly de-allocate all resources
      glDeleteVertexArrays(1, &rail_vao);
      glDeleteBuffers(1, &rail_vbo);
      glDeleteVertexArrays(1, &plank_vao);
      glDeleteBuffers(1, &plank_vbo);
      glDeleteVertexArrays(1, &skybox_vao);
      glDeleteBuffers(1, &skybox_vbo);
      exit(0); // exit the program
    break;
  }
}

void getPoints()
{
    frame = 0;
    numOfVertices = ((splines[0].numControlPoints-3) * 100) + splines[0].numControlPoints-3;

    P = new glm::fvec3[numOfVertices];  //Positions
    T = new glm::fvec3[numOfVertices];  //Tangents
    N = new glm::fvec3[numOfVertices];  //Normals
    B = new glm::fvec3[numOfVertices];  //Bi-normals

    int v_index = 0;

    //Basis matrix
    glm::fmat4 basis;
    basis[0] = glm::fvec4(-s, 2.0 * s, -s, 0.0);
    basis[1] = glm::fvec4(2.0-s, s-3.0, 0.0, 1.0);
    basis[2] = glm::fvec4(s-2.0, 3.0-(2.0*s), s, 0.0);
    basis[3] = glm::fvec4(s, -s, 0.0, 0.0);


    for(int i = 1; i < splines[0].numControlPoints - 2; i++)
    {
        //4 Points
        Point point0 = splines[0].points[i - 1];
        Point point1 = splines[0].points[i];
        Point point2 = splines[0].points[i + 1];
        Point point3 = splines[0].points[i + 2];

        //Control matrix
        glm::fmat3x4 control;
        control[0] = glm::fvec4(point0.x,point1.x,point2.x,point3.x);
        control[1] = glm::fvec4(point0.y,point1.y,point2.y,point3.y);
        control[2] = glm::fvec4(point0.z,point1.z,point2.z,point3.z);

        //Brute force method - evenly spaced u's
      for(float u = 0; u < 1; u += 0.01f)
      {
        glm::fvec4 u_vec(u*u*u, u*u, u, 1);
        glm::fvec3 final_point;

        glm::fvec4 u_diff_vec(3*u*u, 2*u, 1, 0);  //Diff of u matrix
        glm::fvec3 final_tan_point;

        //Compute point
        final_point = u_vec * basis * control;
        P[v_index] = final_point;

        //Compute tangent
        final_tan_point = u_diff_vec * basis * control;
        final_tan_point = glm::normalize(final_tan_point);
        T[v_index] = final_tan_point;

        //Compute Normal
        if(v_index == 0)
          N[v_index] = glm::normalize(glm::cross(glm::vec3(1.0f,1.0f,1.0f), T[v_index]));
        else
          N[v_index] = glm::normalize(glm::cross(B[v_index-1], T[v_index]));

        //Compute Bi-Normal
        B[v_index] = glm::normalize(glm::cross(T[v_index], N[v_index]));

        //Compute Rail vertices at each point
        if(v_index != 0)
          genRails(v_index - 1);

        v_index++;
      }
    }
}

/*Get :
no of splines
Length of spline - num of control points (and type?)
x,y,z control points */
int loadSplines(char * argv) 
{
  char * cName = (char *) malloc(128 * sizeof(char));
  FILE * fileList;
  FILE * fileSpline;
  int iType, i = 0, j, iLength;

  // load the track file 
  fileList = fopen(argv, "r");
  if (fileList == NULL) 
  {
    printf ("can't open file\n");
    exit(1);
  }
  
  // stores the number of splines in a global variable 
  fscanf(fileList, "%d", &numSplines);

  splines = (Spline*) malloc(numSplines * sizeof(Spline));

  // reads through the spline files 
  for (j = 0; j < numSplines; j++) 
  {
    i = 0;
    fscanf(fileList, "%s", cName);
    fileSpline = fopen(cName, "r");

    if (fileSpline == NULL) 
    {
      printf ("can't open file\n");
      exit(1);
    }

    // gets length for spline file
    fscanf(fileSpline, "%d %d", &iLength, &iType);
      cout << "No. of control points: " << iLength << endl;

    // allocate memory for all the points
    splines[j].points = (Point *)malloc(iLength * sizeof(Point));
    splines[j].numControlPoints = iLength;

    int index = 0;
  
    cout << "Spline points: " << endl;
    // saves the data to the struct
    while (fscanf(fileSpline, "%f %f %f", 
	   &splines[j].points[i].x, 
	   &splines[j].points[i].y, 
	   &splines[j].points[i].z) != EOF) 
    {

        cout << "(" << splines[j].points[i].x << "," << splines[j].points[i].y << "," << splines[j].points[i].z << "," << ")" << endl;

        max_coord = max(splines[j].points[i].x, max_coord); //Get the maximum coordinte for the skybox
        max_coord = max(splines[j].points[i].y, max_coord);
        max_coord = max(splines[j].points[i].z, max_coord);
        i++;
    }
  }

  max_coord += 5; //Keep a distance of 5 from skybox and spline

  getPoints();

  free(cName);

  return 0;
}

void skyboxBindProgram()
{
    //use shader
  skybox_pipelineProgram->Bind();

  // get a handle to the modelViewMatrix shader variable
    GLint h_modelViewMatrix = glGetUniformLocation(skybox_program, "modelViewMatrix");
    float m[16];
    skybox_matrix -> SetMatrixMode(OpenGLMatrix::ModelView);
    skybox_matrix->GetMatrix(m);  // column-major
    GLboolean isRowMajor = GL_FALSE;
    glUniformMatrix4fv(h_modelViewMatrix, 1, isRowMajor, m);  //update shader's model view matrix

    // get a handle to the projectionMatrix shader variable
    GLint h_projectionMatrix = glGetUniformLocation(program, "projectionMatrix");
    float p[16]; // column-major
    skybox_matrix -> SetMatrixMode(OpenGLMatrix::Projection);
    skybox_matrix->GetMatrix(p);
    isRowMajor = GL_FALSE;
    glUniformMatrix4fv(h_projectionMatrix, 1, isRowMajor, p); //update shader's projection matrix
}

void bindProgram()
{
    //use shader
    pipelineProgram->Bind();
    
    // get a handle to the modelViewMatrix shader variable
    GLint h_modelViewMatrix = glGetUniformLocation(program, "modelViewMatrix");
    float m[16];
    spline_matrix -> SetMatrixMode(OpenGLMatrix::ModelView);
    spline_matrix->GetMatrix(m);  // column-major
    GLboolean isRowMajor = GL_FALSE;
    glUniformMatrix4fv(h_modelViewMatrix, 1, isRowMajor, m);  //update shader's model view matrix
    
    // get a handle to the projectionMatrix shader variable
    GLint h_projectionMatrix = glGetUniformLocation(program, "projectionMatrix");
    float p[16]; // column-major
    spline_matrix -> SetMatrixMode(OpenGLMatrix::Projection);
    spline_matrix->GetMatrix(p);
    isRowMajor = GL_FALSE;
    glUniformMatrix4fv(h_projectionMatrix, 1, isRowMajor, p); //update shader's projection matrix
}

//set colors for the rail and planks
void setColor(float r,float g,float b)
{
    //use shader
    pipelineProgram->Bind();
    
    // get a handle to the modelViewMatrix shader variable
    GLint color = glGetUniformLocation(program, "color");
    glUniform3f(color, r,g,b);  //update shader's color value
}

//Update matrices with all transformations performed
void skybox_transformations()
{
    cameraPos = P[frame]; //Position
    targetPos = T[frame]; //Tangent
    upVec = N[frame]; //Normal
  
  //Model View
  skybox_matrix -> SetMatrixMode(OpenGLMatrix::ModelView);
  skybox_matrix -> LoadIdentity(); 
  
  skybox_matrix->LookAt(cameraPos.x + upVec.x, cameraPos.y + upVec.y, cameraPos.z + upVec.z, 
                        cameraPos.x + targetPos.x, cameraPos.y + targetPos.y, cameraPos.z + targetPos.z, 
                        upVec.x, upVec.y, upVec.z);
  

    skybox_matrix -> SetMatrixMode(OpenGLMatrix::Projection);
    skybox_matrix -> LoadIdentity();
    skybox_matrix -> Perspective(45.0, 1280/720, 0.1, 1000.0);
}

//Update matrices with all transformations performed
void transformations()
{
    cameraPos = P[frame];  //Position
    targetPos = T[frame]; //Tangent
    upVec = N[frame]; //Normal
    
    //Model View
    spline_matrix -> SetMatrixMode(OpenGLMatrix::ModelView);
    spline_matrix -> LoadIdentity();
    
    spline_matrix->LookAt(cameraPos.x + upVec.x, cameraPos.y + upVec.y, cameraPos.z + upVec.z, 
                          cameraPos.x + targetPos.x, cameraPos.y + targetPos.y, cameraPos.z + targetPos.z, 
                          upVec.x, upVec.y, upVec.z);
    
    
    //Projections
    spline_matrix -> SetMatrixMode(OpenGLMatrix::Projection);
    spline_matrix -> LoadIdentity();
    spline_matrix -> Perspective(45.0, 1280/720, 0.1, 1000.0); //incorrect perspective view? how?
}

void displaySkybox()
{
    skyboxBindProgram();  //bind shader variables
    skybox_transformations(); //update transformation matrices
    
    glBindVertexArray(skybox_vao); // bind the VAO
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture); //bind the texture
    int first = 0;
    int count = 36;
    glDrawArrays(GL_TRIANGLES, first, count);
    glBindVertexArray(0); // unbind the VAO
}

void displaySpline()
{
    transformations();  //update matrices
    bindProgram();  //bind shader variables
    
    //Rails
    glBindVertexArray(rail_vao); // bind the VAO
    setColor(0.5f,0.5f,0.5f); //grey color
    int first = 0;
    int count = rail.size()/3;
    glDrawArrays(GL_TRIANGLES, first, count);
    glBindVertexArray(0); // unbind the VAO

    //Planks
    glBindVertexArray(plank_vao); // bind the VAO
    setColor(0.4850f,0.3250f,0.19f); //Brown color
    first = 0;
    count = plank.size()/3;
    glDrawArrays(GL_TRIANGLES, first, count);
    glBindVertexArray(0); // unbind the VAO
}

void displayFunc()
{
    //Keep incrementing the frame no. to be displayed and restart at 0
    if(frame < numOfVertices)
      frame++;
    else
      frame = 0;

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

    displaySkybox();
    displaySpline();
    
    glutSwapBuffers(); //double - buffered
}

void idleFunc()
{
  if(screenshot_iteration < 1000)  //max of 600 screenshot
  {
    time_now = glutGet(GLUT_ELAPSED_TIME);  //number of milliseconds since the call to glutInit
    if(time_now - prev_time > 66) //15fps take screenshot if 66ms has passed since last capture
    {
      stringstream ss;
      ss << std::setw(3) << std::setfill('0') << screenshot_iteration;  //add leading zeros
      string iter = ss.str();
      prev_time = time_now;

      saveScreenshot(iter + ".jpg");  //take a screenshot
      screenshot_iteration++;
    }
  }
    glutPostRedisplay();
}

/* on Window reshape */
void reshapeFunc(int w, int h)
{
    glViewport(0, 0, w, h);
    
    //update Projection matrices
    GLfloat aspect = (GLfloat)windowWidth/(GLfloat)windowHeight;

    spline_matrix -> SetMatrixMode(OpenGLMatrix::Projection);
    spline_matrix -> LoadIdentity();
    spline_matrix -> Perspective(45.0, aspect, 0.1, 1000.0);
    spline_matrix -> SetMatrixMode(OpenGLMatrix::ModelView);

    skybox_matrix -> SetMatrixMode(OpenGLMatrix::Projection);
    skybox_matrix -> LoadIdentity();
    skybox_matrix -> Perspective(45.0, aspect, 0.1, 1000.0);
    skybox_matrix -> SetMatrixMode(OpenGLMatrix::ModelView);
}

/* Initialize the skybox textures */
void initTexture()
{
  glGenTextures(1, &texture);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_CUBE_MAP, texture); //Bind texture
  
  // Load and generate the texture
  int width, height;
    for(int i = 0; i < 6; i++)
    {
      ImageIO *textureImage = new ImageIO();
      if (textureImage->loadTGA(texture_faces[i]) != ImageIO::OK)
        {
          cout << "Error reading image " << endl;
          exit(EXIT_FAILURE);
        }

        //texture read in the opposite direction so rotate array
        if(i!=2 || i!=3)  //except for top and bottom faces
          textureImage -> Rotate();

      unsigned char* image = textureImage -> getPixels();
      width = textureImage -> getWidth();
      height = textureImage -> getHeight();
      
      glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image); //set texture image for each of the faces
    }

    // Set the texture filtering options
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);  

    glBindTexture(GL_TEXTURE_CUBE_MAP, 0); //Unbind texture
}

//initialize VBO & VAO
void initVBO()
{
    //Rail VAO & VBO
    glGenVertexArrays(1, &rail_vao);
    glBindVertexArray(rail_vao); // bind the VAO
    
    glGenBuffers(1, &rail_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, rail_vbo);
    glBufferData(GL_ARRAY_BUFFER, rail.size() * sizeof(float),rail.data(), GL_STATIC_DRAW);

    GLuint loc = glGetAttribLocation(program,"position");
    glEnableVertexAttribArray(loc);
    const void * offset = (const void*) 0;
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, offset);
    
    glBindVertexArray(0); //unbind VAO

    //Plank VAO & VBO
    glGenVertexArrays(1, &plank_vao);
    glBindVertexArray(plank_vao); // bind the VAO
    
    glGenBuffers(1, &plank_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, plank_vbo);
    glBufferData(GL_ARRAY_BUFFER, plank.size() * sizeof(float),plank.data(), GL_STATIC_DRAW);

    loc = glGetAttribLocation(program,"position");
    glEnableVertexAttribArray(loc);
    offset = (const void*) 0;
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, offset);
    
    glBindVertexArray(0); //unbind VAO

    //SkyBox
    glGenVertexArrays(1, &skybox_vao);
    glBindVertexArray(skybox_vao); // bind the VAO
    
    glGenBuffers(1, &skybox_vbo);
    glBindBuffer(GL_ARRAY_BUFFER,skybox_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skybox_vertices),&skybox_vertices,GL_STATIC_DRAW);
  
    loc = glGetAttribLocation(skybox_program,"position");
    glEnableVertexAttribArray(loc);
    offset = (const void*) 0;
    glVertexAttribPointer(loc, 3, GL_FLOAT, GL_FALSE, 0, offset);
    
    glBindVertexArray(0); // Unbind VAO

}

/* Specify different pipelines for the skybox and the rollerCoaster */
void initPipelineProgram()
{
    //shader stuff
    pipelineProgram = new BasicPipelineProgram();
    pipelineProgram->Init("../openGLHelper-starterCode");
    program = pipelineProgram->GetProgramHandle();

    skybox_pipelineProgram = new BasicPipelineProgram();
    skybox_pipelineProgram->InitSkyBox("../openGLHelper-starterCode");
    skybox_program = skybox_pipelineProgram->GetProgramHandle();
}

/* Specify different transformation matrices for the skybox and the rollerCoaster */
void initMatrices()
{
    spline_matrix = new OpenGLMatrix();
    skybox_matrix = new OpenGLMatrix();
}

/* Specify the skybox as a cube within which the roller coaster is contained */
void setSkyboxCoords()
{
  GLfloat vertices[] = 
  { 
    //Front Face       
    -max_coord,  max_coord, -max_coord,
    -max_coord, -max_coord, -max_coord,
     max_coord, -max_coord, -max_coord,
     max_coord, -max_coord, -max_coord,
     max_coord,  max_coord, -max_coord,
    -max_coord,  max_coord, -max_coord,

    //Left Face
    -max_coord, -max_coord,  max_coord,
    -max_coord, -max_coord, -max_coord,
    -max_coord,  max_coord, -max_coord,
    -max_coord,  max_coord, -max_coord,
    -max_coord,  max_coord,  max_coord,
    -max_coord, -max_coord,  max_coord,

    //Right Face
     max_coord, -max_coord, -max_coord,
     max_coord, -max_coord,  max_coord,
     max_coord,  max_coord,  max_coord,
     max_coord,  max_coord,  max_coord,
     max_coord,  max_coord, -max_coord,
     max_coord, -max_coord, -max_coord,

     //Back Face
    -max_coord, -max_coord,  max_coord,
    -max_coord,  max_coord,  max_coord,
     max_coord,  max_coord,  max_coord,
     max_coord,  max_coord,  max_coord,
     max_coord, -max_coord,  max_coord,
    -max_coord, -max_coord,  max_coord,

    //Top Face
    -max_coord,  max_coord, -max_coord,
     max_coord,  max_coord, -max_coord,
     max_coord,  max_coord,  max_coord,
     max_coord,  max_coord,  max_coord,
    -max_coord,  max_coord,  max_coord,
    -max_coord,  max_coord, -max_coord,

    //Bottom Face
    -max_coord, -max_coord, -max_coord,
    -max_coord, -max_coord,  max_coord,
     max_coord, -max_coord, -max_coord,
     max_coord, -max_coord, -max_coord,
    -max_coord, -max_coord,  max_coord,
     max_coord, -max_coord,  max_coord
   };
   copy(begin(vertices), end(vertices), begin(skybox_vertices));
}

/* initialize the scene */
void initScene(int argc, char *argv[])
{
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glEnable(GL_DEPTH_TEST);
    
    // load the splines from the provided filename (no. of splines, num of control points & control points)
    loadSplines(argv[1]);
    printf("Loaded %d spline(s).\n", numSplines);
    for(int i=0; i<numSplines; i++)
        printf("Num control points in spline %d: %d.\n", i, splines[i].numControlPoints);
    
    //Initialize coordinates for the skybox based on max coordinate obtained from spline file
    setSkyboxCoords();

    //initialize the skybox and spline transformation matrices
    initMatrices();

    //initialize the skybox and spline pipeline programs
    initPipelineProgram();

    //Initialize the VBO & VAO for the Skybox and Spline
    initVBO();

    //initialize the skybox textures
    initTexture();
}

int main (int argc, char ** argv)
{
    //requires track file inpute
    if (argc<2)
    {  
      printf ("usage: %s <trackfile>\n", argv[0]);
      exit(0);
    }

    cout << "Initializing GLUT..." << endl;
    glutInit(&argc,argv);
    
    cout << "Initializing OpenGL..." << endl;
    
    #ifdef __APPLE__
        glutInitDisplayMode(GLUT_3_2_CORE_PROFILE | GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
    #else
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
    #endif
    
    glutInitWindowSize(windowWidth, windowHeight);
    glutInitWindowPosition(0, 0);
    glutCreateWindow(windowTitle);
    
    cout << "OpenGL Version: " << glGetString(GL_VERSION) << endl;
    cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << endl;
    cout << "Shading Language Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;
    
    // tells glut to use a particular display function to redraw
    glutDisplayFunc(displayFunc);
    // perform animation inside idleFunc
    glutIdleFunc(idleFunc);
    // callback for resizing the window
    glutReshapeFunc(reshapeFunc);
    // callback for pressing the keys on the keyboard
    glutKeyboardFunc(keyboardFunc);

    
    // init glew
    #ifdef __APPLE__
        // nothing is needed on Apple
    #else
        // Windows, Linux
        GLint result = glewInit();
        if (result != GLEW_OK)
        {
            cout << "error: " << glewGetErrorString(result) << endl;
            exit(EXIT_FAILURE);
        }
    #endif
    
    // do initialization
    initScene(argc, argv);

    // sink forever into the glut loop
    glutMainLoop();

  return 0;
}

