#<fragment shader>

#version 460 core 

layout (location = 0) out vec4 frag_color;

in vec3 v_position;
in vec2 v_texcoord;
in vec3 v_normal;

uniform vec4 color;

const vec3 light_direction = normalize(vec3(-1, -1, -1));

void main(){
	frag_color = color;
 }

