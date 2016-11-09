uniform sampler2D tex0;
uniform sampler2D tex1;
uniform float weight;

varying vec2 frag_texcoord;

void main() {
    //vec4 rgba0 = texture2D(tex0, frag_texcoord);
    //vec4 rgba1 = texture2D(tex1, frag_texcoord);

    vec4 rgba0 = texture2D(tex0, gl_TexCoord[0].st);
    vec4 rgba1 = texture2D(tex1, gl_TexCoord[0].st);

    if (rgba0.r == 0.0 && rgba0.g == 0.0 && rgba0.b == 0.0) {
        gl_FragColor = rgba1;
    } else if (rgba1.r == 0.0 && rgba1.g == 0.0 && rgba1.b == 0.0) {
        gl_FragColor = rgba0;
    } else {
        gl_FragColor = rgba0 * weight + rgba1 * (1.0 - weight);
        //gl_FragColor = vec4(1,1,1,1);
    }
}
