#version 150

in vec3 texCoord;
out vec4 c;
uniform samplerCube ourTexture;

void main()
{
  // compute the final pixel color
  c = texture(ourTexture, texCoord);
}

