attribute vec3 vPosition;
attribute vec2 texcoord;
//uniform varibles
uniform mat4 mv_mat;
uniform mat4 mp_mat;
uniform float weight;

varying vec2 frag_texcoord;
varying float frag_weight;

void main(){
     gl_Position = mp_mat * mv_mat * vec4(vPosition, 1.0);
     frag_texcoord = texcoord;
     frag_weight = weight;
}
