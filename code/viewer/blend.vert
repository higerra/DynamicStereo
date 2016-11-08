attribute vec3 vPosition;
attribute vec2 texcoord;

uniform mat4 mv_mat;
uniform mat4 mp_mat;

varying vec2 frag_texcoord;
void main(void)
{
    //gl_Position = mp_mat * mv_mat * vec4(vPosition,1.0);

    gl_Position = ftransform();
    gl_TexCoord[0] = gl_MultiTexCoord0;
    //frag_texcoord = texcoord;
}
