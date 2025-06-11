#<fragment shader>
#version 460 core 

layout (location = 0) out vec4 frag_color;

in vec3 v_position;
in vec2 v_texcoord;

uniform sampler3D source_texture;
uniform vec3 texture_resolution;
uniform int render_depth;

void main(){
    float density = texture(source_texture, vec3(v_texcoord, render_depth / texture_resolution.z + 0.5 / texture_resolution.z)).a;
    frag_color = vec4(vec3(density) / 4, 1);
}